# level 1 index 89 agent name: KernelAgent O3 Mini High speedup: 3.66x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# C++ Binding Declaration:
# Expose our CUDA kernel function via an extern "C" API.
# -----------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
extern "C" torch::Tensor cumsum_cuda(torch::Tensor input);
"""

# -----------------------------------------------------------------------------
# CUDA Source:
#
# This version launches exactly (ncols/16) threads per block so that each
# thread is active. This removes divergence due to conditional returns.
#
# Each block processes one row. For each thread:
#  a. It loads a contiguous segment of 16 floats from global memory using
#     vectorized float4 loads.
#  b. It computes its local cumulative sum (scan) on the 16 elements.
#  c. Then, a warp‐level inclusive scan is performed using __shfl_sync and the
#     active warp mask (__activemask()). For partial warps (in the last warp)
#     the last thread is determined dynamically.
#  d. Shared memory is used to accumulate per‐warp totals,
#     allowing each thread to add the proper offset.
#  e. Finally, the corrected results are stored back using vectorized float4 stores.
#
# Note: The kernel expects that ncols is divisible by 16.
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ELEMS_PER_THREAD 16

// Optimized cumulative sum kernel specialized for 2D input.
// Each block processes one row, and we launch exactly (ncols/ELEMS_PER_THREAD) threads per block.
__global__ void cumsum_block_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int nrows, int ncols, int active_threads) {
    // Each thread processes ELEMS_PER_THREAD elements.
    const int row = blockIdx.x;  // one block per row
    const int tid = threadIdx.x; // valid in [0, active_threads)

    // Compute start index for this thread’s segment.
    int start = tid * ELEMS_PER_THREAD;
    int vec_start = start / 4;  // using float4: 4 floats at a time

    // Vectorized load: load 16 floats (as 4 float4 vectors).
    const float4* in_ptr = reinterpret_cast<const float4*>(input + row * ncols);
    float4 v0 = in_ptr[vec_start + 0];
    float4 v1 = in_ptr[vec_start + 1];
    float4 v2 = in_ptr[vec_start + 2];
    float4 v3 = in_ptr[vec_start + 3];

    // Compute the local cumulative sum for this thread's segment.
    float r0 = v0.x;
    float r1 = r0 + v0.y;
    float r2 = r1 + v0.z;
    float r3 = r2 + v0.w;

    float r4 = r3 + v1.x;
    float r5 = r4 + v1.y;
    float r6 = r5 + v1.z;
    float r7 = r6 + v1.w;

    float r8  = r7 + v2.x;
    float r9  = r8 + v2.y;
    float r10 = r9 + v2.z;
    float r11 = r10 + v2.w;

    float r12 = r11 + v3.x;
    float r13 = r12 + v3.y;
    float r14 = r13 + v3.z;
    float r15 = r14 + v3.w;  // cumulative sum over the 16 elements

    // Save the total of this thread's segment.
    float thread_total = r15;

    // -----------------------------------------------------------------------------
    // Warp‑level Inclusive Scan:
    // We use __activemask() to compute the warp mask so that for partial warps
    // the inactive threads are automatically excluded.
    // -----------------------------------------------------------------------------
    unsigned int mask = __activemask();
    int lane = tid & 31;  // lane index within the warp

    // In-warp inclusive scan. We unroll the loop.
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float tmp = __shfl_up_sync(mask, thread_total, offset);
        if (lane >= offset) {
            thread_total += tmp;
        }
    }
    // thread_total now holds the inclusive sum for this thread.
    // To get the exclusive result for the thread, subtract its own local sum.
    float warp_exclusive = (lane == 0) ? 0.0f : (thread_total - r15);

    // -----------------------------------------------------------------------------
    // Determine the last active lane in the current warp.
    // For a full warp, this is lane 31; for a partial warp, it is computed by:
    //   lane_last = (number of threads in this warp) - 1.
    // -----------------------------------------------------------------------------
    int warp_start = (tid >> 5) * 32;
    int warp_end = ((warp_start + 32) > active_threads) ? active_threads : (warp_start + 32);
    int lane_last = warp_end - warp_start - 1;
    // Obtain the warp total (inclusive scan of the last active thread).
    float warp_total = __shfl_sync(mask, thread_total, lane_last);

    // -----------------------------------------------------------------------------
    // Inter‑warp dependency via shared memory.
    // Each warp’s last active thread writes its total into shared memory.
    // Then, each thread aggregates the totals of all previous warps.
    // -----------------------------------------------------------------------------
    extern __shared__ float warp_sums[];
    int warp_id = tid >> 5;
    if (lane == lane_last) {
        warp_sums[warp_id] = warp_total;
    }
    __syncthreads();

    float warp_prefix = 0.0f;
    #pragma unroll
    for (int i = 0; i < warp_id; i++) {
        warp_prefix += warp_sums[i];
    }

    float add_val = warp_exclusive + warp_prefix;

    // -----------------------------------------------------------------------------
    // Add the computed offset to each local element’s cumulative sum.
    // -----------------------------------------------------------------------------
    float o0  = r0  + add_val;
    float o1  = r1  + add_val;
    float o2  = r2  + add_val;
    float o3  = r3  + add_val;
    float o4  = r4  + add_val;
    float o5  = r5  + add_val;
    float o6  = r6  + add_val;
    float o7  = r7  + add_val;
    float o8  = r8  + add_val;
    float o9  = r9  + add_val;
    float o10 = r10 + add_val;
    float o11 = r11 + add_val;
    float o12 = r12 + add_val;
    float o13 = r13 + add_val;
    float o14 = r14 + add_val;
    float o15 = r15 + add_val;

    // -----------------------------------------------------------------------------
    // Vectorized store: write back 16 floats as 4 float4 vectors.
    // -----------------------------------------------------------------------------
    float4* out_ptr = reinterpret_cast<float4*>(output + row * ncols);
    int vec_out_index = vec_start;
    out_ptr[vec_out_index + 0] = make_float4(o0, o1, o2, o3);
    out_ptr[vec_out_index + 1] = make_float4(o4, o5, o6, o7);
    out_ptr[vec_out_index + 2] = make_float4(o8, o9, o10, o11);
    out_ptr[vec_out_index + 3] = make_float4(o12, o13, o14, o15);
}

extern "C" torch::Tensor cumsum_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "cumsum_cuda: input tensor must be 2D");
    int nrows = input.size(0);
    int ncols = input.size(1);
    // Ensure the number of columns is divisible by 16.
    TORCH_CHECK(ncols % (ELEMS_PER_THREAD) == 0, "cumsum_cuda: ncols must be divisible by 16 for optimized kernel");
    auto output = torch::empty_like(input);

    // Compute the number of threads per block: one thread per 16-element segment.
    int active_threads = ncols / ELEMS_PER_THREAD;
    // Determine shared memory size: one float per warp.
    int num_warps = (active_threads + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(float);

    // Launch one block per row with exactly active_threads threads per block.
    dim3 grid(nrows);
    dim3 block(active_threads);
    cumsum_block_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        nrows, ncols, active_threads
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
    }
    return output;
}
"""

# -----------------------------------------------------------------------------
# Compile the Inline CUDA Extension.
# -----------------------------------------------------------------------------
cumsum_module = load_inline(
    name="cumsum_extension_optimized",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cumsum_cuda"],
    verbose=True,
)

# -----------------------------------------------------------------------------
# Optimized Model Class
#
# This Model computes the cumulative sum along dimension 1 using the ultra‑optimized
# CUDA kernel defined above. The external interface is identical to the original Model.
#
# Note: Only cumulative sum along dimension 1 is supported, and the number of columns 
# must be divisible by 16.
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix scan) operation along 
    dimension 1 using an ultra‑optimized CUDA kernel.
    
    Note: Only cumulative sum along dimension 1 is supported, and ncols (number of columns)
          must be divisible by 16.
    """
    def __init__(self, dim):
        """
        Initialize the Model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
                       Only dim == 1 is supported.
        """
        super(Model, self).__init__()
        self.dim = dim
        if self.dim != 1:
            raise ValueError("Optimized Model only supports cumulative sum along dimension 1.")

    def forward(self, x):
        """
        Forward pass: Computes the cumulative sum along dimension 1.
        The input must be a 2D tensor with number of columns divisible by 16.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_cols).

        Returns:
            torch.Tensor: Tensor of the same shape as x, with cumulative sum computed along dim 1.
        """
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D for this optimized cumulative sum implementation.")
        return cumsum_module.cumsum_cuda(x)

# -----------------------------------------------------------------------------
# Helper Functions (interface matching the original code):
# -----------------------------------------------------------------------------
batch_size = 128
input_shape = (4000,)  # Example shape; note that 4000 is divisible by 16.
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Model.

    Returns:
        list: A list containing one randomly generated tensor of shape
              (batch_size, *input_shape).
    """
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    """
    Provides the initialization parameters for the Model.

    Returns:
        list: A list containing the 'dim' parameter for model initialization.
    """
    return [dim]

# -----------------------------------------------------------------------------
# Example Usage / Sanity Check.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(dim=1).to(device)
    # Create an input tensor of shape (128, 4000); note that 4000 is divisible by 16.
    x = torch.randn(batch_size, input_shape[0], device=device, dtype=torch.float32)
    y = model(x)
    # Validate against PyTorch's built‑in cumulative sum.
    y_ref = torch.cumsum(x, dim=1)
    if torch.allclose(y, y_ref, atol=1e-5):
        print("Ultra‑optimized cumsum kernel matches torch.cumsum!")
    else:
        print("Mismatch between optimized kernel and torch.cumsum!")
