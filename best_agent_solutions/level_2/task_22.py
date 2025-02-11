# level 2 index 22 agent name: KernelAgent O3 Mini High speedup: 4.15x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# The CUDA kernel implements a fused operator that, for each row of the input matrix (the
# output of the linear layer), does the following:
#  1. Multiply each element by a pre-computed combined_factor (which equals 2.0 * scale_factor).
#  2. Clamp each result between clamp_min and clamp_max.
#  3. Compute a numerically-stable row-wise logsumexp reduction across all elements.
#  4. Compute softplus = log(1+exp(lse)).
#  5. Compute the final activated value as (lse^2) * tanhf(softplus).
#
# Since mish(x) = x * tanh(softplus(x)), note that:
#    x * mish(x) = x^2 * tanh(softplus(x))
#
# Thus the kernel computes lse = logsumexp(clamped(row)) and returns
#   final_val = lse^2 * tanhf(softplus(lse)).
#
# For maximum performance the kernel:
#   - Loads 4 floats at a time (assuming hidden_size is divisible by 4).
#   - Uses warp shuffle instructions for intra-warp reduction.
#   - Uses shared memory for inter-warp reduction before finalizing the output.
cuda_source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <float.h>

#define WARP_SIZE 32

// Clamps value v between lo and hi.
__device__ __forceinline__ float clamp_val(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// Update a logsumexp accumulator given a new value c.
__device__ __forceinline__ void lse_pair_update(float c, float &m, float &s) {
    if (c > m) {
        s = s * __expf(m - c) + 1.0f;
        m = c;
    } else {
        s += __expf(c - m);
    }
}

// Combine two logsumexp accumulators (m1,s1) and (m2,s2) in a numerically stable way.
__device__ __forceinline__ void lse_pair_combine(float m1, float s1, float m2, float s2, float &m_out, float &s_out) {
    float m_new = fmaxf(m1, m2);
    if (m1 >= m2)
        s_out = s1 + s2 * __expf(m2 - m1);
    else
        s_out = s2 + s1 * __expf(m1 - m2);
    m_out = m_new;
}

extern "C" {

// Fused CUDA kernel. Each block processes one row of linear_output (shape: (batch_size, hidden_size)).
__global__ __launch_bounds__(256)
void fused_logsumexp_mish_kernel_singlepass(
    const float* __restrict__ linear_output,  // shape: (batch_size, hidden_size)
    float* __restrict__ out,                    // shape: (batch_size, 1)
    int hidden_size,
    float combined_factor,
    float clamp_min,
    float clamp_max)
{
    const int row = blockIdx.x;  // one block per row.
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    // Process data as float4 for vectorized memory access.
    int vec_length = hidden_size >> 2;  // hidden_size / 4
    const float4* row_vec = reinterpret_cast<const float4*>(linear_output);

    // Each thread maintains its own local logsumexp accumulator.
    float local_m = -FLT_MAX;
    float local_s = 0.0f;

    // Strided loop over the vectorized elements.
    #pragma unroll
    for (int j = tid; j < vec_length; j += blockDim.x) {
        float4 vec = row_vec[row * vec_length + j];

        float tmp = vec.x * combined_factor;
        float c = clamp_val(tmp, clamp_min, clamp_max);
        lse_pair_update(c, local_m, local_s);

        tmp = vec.y * combined_factor;
        c = clamp_val(tmp, clamp_min, clamp_max);
        lse_pair_update(c, local_m, local_s);

        tmp = vec.z * combined_factor;
        c = clamp_val(tmp, clamp_min, clamp_max);
        lse_pair_update(c, local_m, local_s);

        tmp = vec.w * combined_factor;
        c = clamp_val(tmp, clamp_min, clamp_max);
        lse_pair_update(c, local_m, local_s);
    }

    // Intra-warp reduction using shuffle instructions.
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float m_sh = __shfl_down_sync(0xffffffff, local_m, offset);
        float s_sh = __shfl_down_sync(0xffffffff, local_s, offset);
        float combined_m, combined_s;
        lse_pair_combine(local_m, local_s, m_sh, s_sh, combined_m, combined_s);
        local_m = combined_m;
        local_s = combined_s;
    }

    // Each warp writes its reduced result to shared memory.
    __shared__ float shared_m[32];
    __shared__ float shared_s[32];
    if (lane == 0) {
        shared_m[warp_id] = local_m;
        shared_s[warp_id] = local_s;
    }
    __syncthreads();

    // Let the first warp reduce the results from all warps.
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float block_m = (tid < num_warps) ? shared_m[tid] : -FLT_MAX;
    float block_s = (tid < num_warps) ? shared_s[tid] : 0.0f;
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float m_sh = __shfl_down_sync(0xffffffff, block_m, offset);
        float s_sh = __shfl_down_sync(0xffffffff, block_s, offset);
        float combined_m, combined_s;
        lse_pair_combine(block_m, block_s, m_sh, s_sh, combined_m, combined_s);
        block_m = combined_m;
        block_s = combined_s;
    }

    // The first thread writes the final output.
    if (tid == 0) {
        float lse = block_m + __logf(block_s);
        float softplus = __logf(1.0f + __expf(lse));
        // Final value: (lse^2) * tanh(softplus), which is equivalent to lse * mish(lse)
        float final_val = (lse * lse) * tanhf(softplus);
        out[row] = final_val;
    }
}

at::Tensor fused_logsumexp_mish_cuda_fast(at::Tensor linear_output,
                                          int hidden_size,
                                          float combined_factor,
                                          float clamp_min,
                                          float clamp_max) {
    const int batch_size = linear_output.size(0);
    auto out = at::empty({batch_size, 1}, linear_output.options());
    int vec_length = hidden_size >> 2;
    // Choose the number of threads: use 256 if possible.
    int threads = (vec_length < 256 ? vec_length : 256);
    int blocks = batch_size;
    fused_logsumexp_mish_kernel_singlepass<<<blocks, threads>>>(
        linear_output.data_ptr<float>(),
        out.data_ptr<float>(),
        hidden_size,
        combined_factor,
        clamp_min,
        clamp_max);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return out;
}

} // extern "C"
'''

cpp_source = r'''
extern "C" {
  at::Tensor fused_logsumexp_mish_cuda_fast(const at::Tensor linear_output, int hidden_size, float combined_factor, float clamp_min, float clamp_max);
}
'''

# Compile and load the CUDA extension.
fused_module = load_inline(
    name="fused_logsumexp_mish_fused_fast",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_logsumexp_mish_cuda_fast"],
    verbose=False,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class Model(nn.Module):
    """
    Optimized Model that fuses multiple elementwise operations:
       1. Instead of computing (x * scale_factor) followed by (x + x),
          we compute x * (2.0 * scale_factor).
       2. Instead of separately clamping, computing row-wise logsumexp, and
          applying x * mish(x),
          we fuse these into a single CUDA kernel that computes:
             final_val = (lse^2) * tanhf(softplus(lse))
          which is mathematically equivalent to lse * mish(lse), where
             mish(z) = z * tanh(softplus(z)).
             
    The module preserves the original interface and behavior.
    
    Note: We require that hidden_size is divisible by 4 for efficient vectorized loads.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        # Using nn.Linear; its parameters are initialized in the standard way.
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.hidden_size = hidden_size
        # Check for vectorized load compatibility.
        if hidden_size % 4 != 0:
            raise ValueError("hidden_size must be divisible by 4 for optimized kernel.")

    def forward(self, x):
        # Apply the matrix multiplication.
        x = self.matmul(x)  # shape: (batch_size, hidden_size)
        # Fuse the two operations: (x * scale_factor) + x  ===  x * (2.0 * scale_factor)
        combined_factor = 2.0 * self.scale_factor

        # Ensure the tensor is contiguous for vectorized loads in our CUDA kernel.
        if not x.is_contiguous():
            x = x.contiguous()

        out = fused_module.fused_logsumexp_mish_cuda_fast(
            x,
            self.hidden_size,
            combined_factor,
            self.clamp_min,
            self.clamp_max
        )
        return out

# Global parameters.
batch_size = 128
input_size = 512
hidden_size = 1024  # Must be divisible by 4.
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size, dtype=torch.float32)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(*get_init_inputs()).to(device)
    x = get_inputs()[0].to(device)
    # Warm-up for CUDA (optional but can reduce startup overhead in timing tests)
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    y = model(x)
    print("Output:\n", y)
