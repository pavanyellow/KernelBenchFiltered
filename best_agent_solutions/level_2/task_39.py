# level 2 index 39 agent name: KernelAgent 4o speedup: 1.32x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel to optimize the batch normalization and scaling
batch_norm_scale_fused_source_optimized = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_scale_fused_kernel_optimized(
    const float* __restrict__ x, 
    float* __restrict__ out,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    const float* __restrict__ scale,
    int batch_size, 
    int out_features, 
    float eps) {
    
    extern __shared__ float shared_data[];
    float* mean = shared_data;
    float* variance = shared_data + out_features;

    int tid = threadIdx.x;
    int feature_idx = blockIdx.x * blockDim.x + tid;

    if (feature_idx < out_features) {
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;

        // Compute mean and variance
        for (int i = 0; i < batch_size; ++i) {
            float val = x[i * out_features + feature_idx] * scale[feature_idx];
            local_sum += val;
            local_sum_sq += val * val;
        }

        mean[feature_idx] = local_sum / batch_size;
        variance[feature_idx] = local_sum_sq / batch_size - mean[feature_idx] * mean[feature_idx];

        float inv_var = rsqrtf(variance[feature_idx] + eps);
        float gamma_val = gamma[feature_idx];
        float beta_val = beta[feature_idx];

        for (int i = 0; i < batch_size; ++i) {
            int index = i * out_features + feature_idx;
            out[index] = ((x[index] * scale[feature_idx] - mean[feature_idx]) * inv_var) * gamma_val + beta_val;
        }
    }
}

torch::Tensor batch_norm_scale_fused_cuda_optimized(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor scale,
    float eps) {

    auto batch_size = x.size(0);
    auto out_features = x.size(1);

    auto options = x.options();
    auto out = torch::empty_like(x, options);

    const int block_size = 64;  // Trying to find more optimal number of threads per block
    const int num_blocks = (out_features + block_size - 1) / block_size;

    batch_norm_scale_fused_kernel_optimized<<<num_blocks, block_size, 2 * out_features * sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        scale.data_ptr<float>(),
        batch_size, out_features, eps
    );

    return out;
}
"""

batch_norm_scale_fused_cpp_source_optimized = """
torch::Tensor batch_norm_scale_fused_cuda_optimized(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor scale,
    float eps);
"""

# Compile the inline CUDA code
batch_norm_scale_fused_native_module_optimized = load_inline(
    name='batch_norm_scale_fused_optimized',
    cpp_sources=batch_norm_scale_fused_cpp_source_optimized,
    cuda_sources=batch_norm_scale_fused_source_optimized,
    functions=['batch_norm_scale_fused_cuda_optimized'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.batch_norm_weight = nn.Parameter(torch.ones(out_features))
        self.batch_norm_bias = nn.Parameter(torch.zeros(out_features))
        self.eps = eps

    def forward(self, x):
        # Perform GEMM using PyTorch's efficient implementation
        x = torch.matmul(x, self.gemm.weight.t()) + self.gemm.bias

        # Fused scaled batch normalization using optimized CUDA
        x = batch_norm_scale_fused_native_module_optimized.batch_norm_scale_fused_cuda_optimized(
            x, self.batch_norm_weight, self.batch_norm_bias, self.scale, self.eps
        )

        return x

# Equivalent functional code to test the above Model class.
batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
