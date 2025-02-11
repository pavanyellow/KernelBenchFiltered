# level 2 index 75 agent name: KernelAgent O3 Mini High speedup: 1.90x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------------------
# Declare the C++ API for the fused CUDA function.
# ------------------------------------------------------------------------------
fused_gn_min_bias_cpp_source = r'''
torch::Tensor fused_gn_min_bias_cuda(torch::Tensor X,
                                       torch::Tensor gn_weight,
                                       torch::Tensor gn_bias,
                                       torch::Tensor bias,
                                       int n, int c,
                                       int group_size,
                                       float eps);
'''

# ------------------------------------------------------------------------------
# Define the CUDA source that implements the fused kernel.
# ------------------------------------------------------------------------------
fused_gn_min_bias_cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

// Fused CUDA kernel: performs group normalization (per group),
// then a fused minimum reduction over all (normalized) channels and
// finally adds a per–channel bias.
// The kernel assumes that:
//   - X has shape (n, c) where each block (indexed by sample) reads one sample.
//   - The channels are split into groups of size "group_size" (num_groups = c/group_size).
//   - The output is a tensor of shape (1, c, n, 1) so that for each sample the
//     global minimum (across groups) is computed and then broadcast to every channel
//     after bias addition.
__global__ void fused_gn_min_bias_kernel(const float * __restrict__ X,
                                           const float * __restrict__ gn_weight,
                                           const float * __restrict__ gn_bias,
                                           const float * __restrict__ bias,
                                           float * __restrict__ output,
                                           int n, int c, int group_size, float eps) {
    // Each block processes one sample.
    int sample_idx = blockIdx.x;
    // Each thread processes one channel.
    int ch = threadIdx.x;
    if (ch >= c)
        return;
  
    // Load this sample's input for channel "ch".
    float x_val = X[sample_idx * c + ch];

    // Determine the group id and lane within that group.
    int group_id = ch / group_size;
    int lane = ch % group_size;
    const unsigned full_mask = 0xffffffffu;

    // Compute the sum and sum-of-squares for group normalization using warp shuffles.
    float group_sum = x_val;
    float group_sumsq = x_val * x_val;
    if (group_size == 32) {
        // Unrolled reduction (common case when out_features=256 and num_groups=8).
        group_sum += __shfl_down_sync(full_mask, group_sum, 16);
        group_sum += __shfl_down_sync(full_mask, group_sum, 8);
        group_sum += __shfl_down_sync(full_mask, group_sum, 4);
        group_sum += __shfl_down_sync(full_mask, group_sum, 2);
        group_sum += __shfl_down_sync(full_mask, group_sum, 1);
  
        group_sumsq += __shfl_down_sync(full_mask, group_sumsq, 16);
        group_sumsq += __shfl_down_sync(full_mask, group_sumsq, 8);
        group_sumsq += __shfl_down_sync(full_mask, group_sumsq, 4);
        group_sumsq += __shfl_down_sync(full_mask, group_sumsq, 2);
        group_sumsq += __shfl_down_sync(full_mask, group_sumsq, 1);
    } else {
        for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
            group_sum  += __shfl_down_sync(full_mask, group_sum, offset);
            group_sumsq += __shfl_down_sync(full_mask, group_sumsq, offset);
        }
    }
    // The first thread in the group now has the complete sum and sumsq.
    float mean = __shfl_sync(full_mask, group_sum, 0) / group_size;
    float variance = __shfl_sync(full_mask, group_sumsq, 0) / group_size - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Use __ldg to load the group norm parameters (assumed read–only).
    float gamma = __ldg(&gn_weight[ch]);
    float beta  = __ldg(&gn_bias[ch]);
    float scale = gamma * inv_std;
    float norm_val = scale * x_val - scale * mean + beta;

    // Next, perform warp-level minimum reduction for just this group.
    float group_min = norm_val;
    if (group_size == 32) {
        group_min = fminf(group_min, __shfl_down_sync(full_mask, group_min, 16));
        group_min = fminf(group_min, __shfl_down_sync(full_mask, group_min, 8));
        group_min = fminf(group_min, __shfl_down_sync(full_mask, group_min, 4));
        group_min = fminf(group_min, __shfl_down_sync(full_mask, group_min, 2));
        group_min = fminf(group_min, __shfl_down_sync(full_mask, group_min, 1));
    } else {
        for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(full_mask, group_min, offset);
            group_min = fminf(group_min, other);
        }
    }
    float warp_min = __shfl_sync(full_mask, group_min, 0);

    // Use dynamic shared memory to collect one min per group.
    int num_groups = c / group_size;
    extern __shared__ float shared_group_mins[];
    if (lane == 0) {
        shared_group_mins[group_id] = warp_min;
    }
    __syncthreads();

    // Thread 0 computes the global minimum across all groups.
    float global_min;
    if (ch == 0) {
        global_min = shared_group_mins[0];
        #pragma unroll
        for (int i = 1; i < num_groups; i++) {
            global_min = fminf(global_min, shared_group_mins[i]);
        }
        // Broadcast the global min.
        shared_group_mins[0] = global_min;
    }
    __syncthreads();
    global_min = shared_group_mins[0];
  
    // Finally, add the per–channel bias (loaded from read–only memory).
    float channel_bias = __ldg(&bias[ch]);
    float result = global_min + channel_bias;
    // Write the result to the output tensor.
    // The output tensor shape is [1, c, n, 1] and we store it as: out[ch * n + sample_idx].
    output[ch * n + sample_idx] = result;
}

torch::Tensor fused_gn_min_bias_cuda(torch::Tensor X,
                                       torch::Tensor gn_weight,
                                       torch::Tensor gn_bias,
                                       torch::Tensor bias,
                                       int n, int c,
                                       int group_size,
                                       float eps) {
    auto output = torch::empty({1, c, n, 1}, X.options());
    int blocks = n;
    int threads = c;
    int shared_mem_size = (c / group_size) * sizeof(float);
    fused_gn_min_bias_kernel<<<blocks, threads, shared_mem_size, c10::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        n, c, group_size, eps
    );
    return output;
}
'''

# ------------------------------------------------------------------------------
# Build the fused CUDA extension.
# ------------------------------------------------------------------------------
fused_gn_min_bias_module = load_inline(
    name="fused_gn_min_bias_module",
    cpp_sources=fused_gn_min_bias_cpp_source,
    cuda_sources=fused_gn_min_bias_cuda_source,
    functions=["fused_gn_min_bias_cuda"],
    verbose=True
)

# ------------------------------------------------------------------------------
# Python wrapper that calls the fused CUDA kernel.
# ------------------------------------------------------------------------------
def fused_gn_min_bias(X, gn_weight, gn_bias, bias, n, c, group_size, eps):
    return fused_gn_min_bias_module.fused_gn_min_bias_cuda(X, gn_weight, gn_bias, bias, n, c, group_size, eps)

# ------------------------------------------------------------------------------
# Optimized Model definition.
# ------------------------------------------------------------------------------
class Model(nn.Module):
    """
    Optimized Model that performs:
      - A GEMM via nn.Linear,
      - Group Normalization via nn.GroupNorm,
      - A fused minimum reduction across channels (after normalization) and
      - Bias addition.
    The module exposes the same interface as the original:
      Model(in_features, out_features, num_groups, bias_shape)
    and produces an output with shape (1, out_features, batch, 1).
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Precompute group size. (out_features must be divisible by num_groups)
        self.group_size = out_features // num_groups

    def forward(self, x):
        # Input x: [batch, in_features]
        gemm_out = self.gemm(x)  # => [batch, out_features]
        gemm_out = gemm_out.contiguous()
        n, c = gemm_out.shape
        # The fused kernel expects bias as a vector of length c.
        return fused_gn_min_bias(gemm_out,
                                 self.group_norm.weight,
                                 self.group_norm.bias,
                                 self.bias.squeeze(),
                                 n, c, self.group_size,
                                 self.group_norm.eps)

# ------------------------------------------------------------------------------
# Helper functions matching the original module interface.
# ------------------------------------------------------------------------------
batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    # Returns a list with one input tensor of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    # Returns a list of arguments to initialize Model: (in_features, out_features, num_groups, bias_shape).
    return [in_features, out_features, num_groups, bias_shape]

# ------------------------------------------------------------------------------
# Simple test run.
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    x = torch.randn(batch_size, in_features, device='cuda')
    model = Model(in_features, out_features, num_groups, bias_shape).cuda()
    
    # Warm-up (trigger lazy compilation).
    y = model(x)
    torch.cuda.synchronize()
    
    import time
    start = time.time()
    for _ in range(1000):
        y = model(x)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Average run time: {(end - start) / 1000 * 1e3:.5f} ms")
    
    # The expected output shape is (1, out_features, batch_size, 1)
    print("Output shape:", y.shape)
