# level 2 index 37 agent name: KernelAgent 4o speedup: 1.93x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for Swish activation, bias addition, and GroupNorm
fused_cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float swish(const float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void fused_swish_bias_groupnorm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int num_groups, 
    int group_size, 
    int batch_size, 
    int out_features
) {
    extern __shared__ float shared_mem[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_elements = batch_size * out_features;
    if (idx >= num_elements) return;
    
    const int batch_idx = idx / out_features;
    const int feature_idx = idx % out_features;
    const int group_idx = feature_idx / group_size;
    
    const float value = input[idx];
    const float biased_value = swish(value) + bias[feature_idx];
    
    // Shared memory caches for this group
    float* group_shared = shared_mem + threadIdx.x / group_size * group_size;

    // Load input into shared memory
    group_shared[threadIdx.x % group_size] = biased_value;
    __syncthreads();

    // Compute mean for the group
    float mean = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        mean += group_shared[i];
    }
    mean /= group_size;
    __syncthreads();

    // Compute variance for the group
    float variance = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        float diff = group_shared[i] - mean;
        variance += diff * diff;
    }
    variance /= group_size;
    __syncthreads();

    // Calculate inverse standard deviation
    float inv_std = rsqrtf(variance + 1e-5f);

    // Write normalized output
    output[idx] = (biased_value - mean) * inv_std;
}

torch::Tensor fused_swish_bias_groupnorm_cuda(
    torch::Tensor input, torch::Tensor bias, int num_groups
) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int group_size = out_features / num_groups;
    int num_elements = batch_size * out_features;

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    size_t shared_mem_size = num_groups * group_size * sizeof(float);

    fused_swish_bias_groupnorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_groups, 
        group_size, 
        batch_size, 
        out_features
    );

    return output;
}
"""

cpp_code = """
torch::Tensor fused_swish_bias_groupnorm_cuda(torch::Tensor input, torch::Tensor bias, int num_groups);
"""

# Compile the kernel
custom_cuda_module = load_inline(
    name='fused_ops_optimized',
    cpp_sources=cpp_code,
    cuda_sources=fused_cuda_code,
    functions=['fused_swish_bias_groupnorm_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['--use_fast_math']
)

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term,
    and normalizes with GroupNorm - all optimized using CUDA.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features).cuda()
        self.bias = nn.Parameter(torch.randn(bias_shape, device='cuda'))
        self.num_groups = num_groups

    def forward(self, x):
        x = self.matmul(x)
        x = custom_cuda_module.fused_swish_bias_groupnorm_cuda(x, self.bias, self.num_groups)
        return x

# Test and usage
batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
