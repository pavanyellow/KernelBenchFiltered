# level 1 index 91 agent name: KernelAgent O3 Mini High speedup: 2.93x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Here we use cub’s block-scan to compute the exclusive prefix sum across the row.
# That lets each thread load a chunk of float4’s, compute its own local sum,
# and then compute the suffix‐scan (i.e. total_sum – prefix) in one fused kernel.
#
# We continue to assume:
#    (1) the tensor is 2D, contiguous, and on CUDA;
#    (2) width (i.e. x.size(1)) is divisible by 4 and the pointers are 16-byte aligned.
#
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#define THREADS_PER_BLOCK 256
#define MAX_CHUNK 1024

// This kernel computes a fused suffix-sum for one row of the input.
// Given an input row, for each element index j it computes:
//     output[j] = sum_{k=j}^{width-1} input[k]
// which is mathematically equivalent to:
//     torch.cumsum(x.flip(1), dim=1).flip(1)
// Assumptions: width is divisible by 4 and input/out pointers are 16-byte aligned.
extern "C"
__global__ void suffix_scan_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int width) {
    // Each block handles one row.
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    // We load data in vectorized float4 loads. (width must be divisible by 4.)
    const int width_v = width >> 2;  // number of float4 segments per row.
    // Divide the work: each thread processes a contiguous chunk of float4 segments.
    int chunk_size = (width_v + num_threads - 1) / num_threads;
    int start = tid * chunk_size;
    
    // Set up pointers for the current row.
    const float* in_row = input + row * width;
    float* out_row = output + row * width;
    
    // Temporary storage in registers for our loaded values.
    // We use a fixed-size buffer (MAX_CHUNK) and assume (chunk_size*4) does not exceed this.
    float vals[MAX_CHUNK];
    int local_count = 0;
    float thread_total = 0.0f;
    
    // Load our chunk using vectorized loads (float4) and accumulate our local sum.
#pragma unroll
    for (int i = 0; i < chunk_size; i++) {
        int idx_v = start + i;
        if (idx_v < width_v) {
            float4 vec = reinterpret_cast<const float4*>(in_row)[idx_v];
            int offset = i * 4;
            vals[offset + 0] = vec.x;
            vals[offset + 1] = vec.y;
            vals[offset + 2] = vec.z;
            vals[offset + 3] = vec.w;
            thread_total += (vec.x + vec.y + vec.z + vec.w);
            local_count += 4;
        }
    }
    
    // Now perform a block-level inclusive scan (using cub::BlockScan) over the thread totals.
    // After the inclusive scan, each thread has the sum of all thread_totals from lane 0 up to its own lane,
    // from which we can compute its exclusive prefix as (inclusive - thread_total).
    typedef cub::BlockScan<float, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    float thread_inclusive = 0.0f;
    BlockScan(temp_storage).InclusiveSum(thread_total, thread_inclusive);
    float thread_exclusive = thread_inclusive - thread_total;
    
    // The total sum for the row is the inclusive sum of the very last thread.
    __shared__ float block_total;
    if (tid == num_threads - 1) {
        block_total = thread_inclusive;
    }
    __syncthreads();
    float total_sum = block_total;
    
    // Each thread now does its own local exclusive scan (in registers) over its loaded values.
    // Then each output element is computed as: total_sum - (global_prefix + local_prefix)
    int start_idx = start << 2;  // equivalent to start * 4
    float running = 0.0f;
#pragma unroll
    for (int i = 0; i < local_count; i++) {
        float prefix = thread_exclusive + running;
        out_row[start_idx + i] = total_sum - prefix;
        running += vals[i];
    }
}
  
// The entry-point wrapper that launches the kernel.
torch::Tensor suffix_scan_forward_cuda(torch::Tensor input) {
    // assume input is a 2D tensor of type float
    int rows = input.size(0);
    int width = input.size(1);
    auto output = torch::empty_like(input);
    
    dim3 grid(rows);
    dim3 block(THREADS_PER_BLOCK);
    suffix_scan_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        width);
    return output;
}
"""

cpp_source = r"""
torch::Tensor suffix_scan_forward_cuda(torch::Tensor input);
"""

# Compile the inline CUDA extension. We use aggressive optimizations plus fast-math flags.
auto_module = load_inline(
    name="suffix_scan_fused_faster",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["suffix_scan_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class Model(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super(Model, self).__init__()
        # This module expects a 2D input with dim==1 for optimized processing.
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fall back to native PyTorch if the target dim isn’t 1, the tensor isn’t 2D,
        # or if the width isn’t divisible by 4 or the memory isn’t 16-byte aligned.
        if self.dim != 1 or x.dim() != 2:
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)
        x = x.contiguous()
        if (x.size(1) & 3) or (x.data_ptr() & 15):
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)
        # Use the optimized CUDA kernel.
        return auto_module.suffix_scan_forward_cuda(x)

def get_init_inputs():
    # The model initialization arguments: an integer dimension (always 1).
    return (1,)

def get_inputs():
    # The input tensor: shape (128, 4000), dtype torch.float32.
    return (torch.randn(128, 4000, dtype=torch.float32).cuda(),)

if __name__ == "__main__":
    model = Model(1).cuda()
    x = torch.randn(128, 4000, dtype=torch.float32).cuda()
    
    # Warm-up run
    y_fast = model(x)
    y_ref = torch.cumsum(x.flip(1), dim=1).flip(1)
    print("Allclose:", torch.allclose(y_fast, y_ref, atol=1e-6))
    
    # Timing test.
    torch.cuda.synchronize()
    import time
    runs = 1000
    start_time = time.time()
    for _ in range(runs):
        y = model(x)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_ms = ((end_time - start_time) / runs) * 1000
    print(f"Average time per run: {avg_ms:.6f} ms")
