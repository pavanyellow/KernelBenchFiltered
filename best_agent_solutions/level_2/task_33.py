# level 2 index 33 agent name: KernelAgent 4o speedup: 1.75x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA source code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scale_norm_kernel(
    const float* input, const float* scale, float* output,
    const float* weight, const float* bias, float* mean, float* var,
    int batch_size, int features, float eps) {

    extern __shared__ float shared[]; // shared memory
    float* shared_sum = shared;
    float* shared_squares = shared + blockDim.x;

    int thread_id = threadIdx.x;
    int feature_idx = blockIdx.x;
    int size = batch_size * features;

    // Initialize shared memory
    shared_sum[thread_id] = 0.0f;
    shared_squares[thread_id] = 0.0f;

    // Step 1: Compute intermediate scale, sum, and squared sum
    for (int i = thread_id; i < batch_size; i += blockDim.x) {
        int feature_offset = i * features + feature_idx;
        if (feature_offset < size) {
            float scaled_input = input[feature_offset] * scale[feature_idx];
            atomicAdd(&shared_sum[thread_id], scaled_input);
            atomicAdd(&shared_squares[thread_id], scaled_input * scaled_input);
            output[feature_offset] = scaled_input;
        }
    }
    __syncthreads();

    // Reduction across block
    if (thread_id == 0) {
        float total_sum = 0.0f;
        float total_squares = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            total_sum += shared_sum[i];
            total_squares += shared_squares[i];
        }
        mean[feature_idx] = total_sum / batch_size;
        var[feature_idx] = total_squares / batch_size - mean[feature_idx] * mean[feature_idx];
    }
    __syncthreads();

    // Step 3: Normalize and apply weight and bias
    float current_mean = mean[feature_idx];
    float current_var = var[feature_idx] + eps;
    float inv_std = rsqrtf(current_var);

    for (int i = thread_id; i < batch_size; i += blockDim.x) {
        int feature_offset = i * features + feature_idx;
        if (feature_offset < size) {
            float norm_val = (output[feature_offset] - current_mean) * inv_std;
            output[feature_offset] = norm_val * weight[feature_idx] + bias[feature_idx];
        }
    }
}

torch::Tensor fused_scale_norm(torch::Tensor input, torch::Tensor scale, torch::Tensor weight, torch::Tensor bias, float eps) {
    int batch_size = input.size(0);
    int features = input.size(1);

    auto output = torch::empty_like(input);
    auto mean = torch::empty({features}, input.options());
    auto var = torch::empty({features}, input.options());

    int block_size = 256;
    int grid_size = features;
    int shared_memory_size = block_size * 2 * sizeof(float);

    fused_scale_norm_kernel<<<grid_size, block_size, shared_memory_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size, features, eps
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_scale_norm(torch::Tensor input, torch::Tensor scale, torch::Tensor weight, torch::Tensor bias, float eps);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name='fused_scale_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_scale_norm'],
    verbose=True
)

class Model(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.eps = eps

    def forward(self, x):
        x = self.gemm(x)
        x = fused_module.fused_scale_norm(x, self.scale, self.bn_weight, self.bn_bias, self.eps)
        return x

# Function to provide inputs
def get_inputs():
    return [torch.randn(128, 1024)]

# Function to provide initialization inputs
def get_init_inputs():
    return [1024, 512, (512,)]
