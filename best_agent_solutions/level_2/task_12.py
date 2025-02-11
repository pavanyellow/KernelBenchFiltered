# level 2 index 12 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.18x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_mul_leakyrelu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float multiplier,
    const float negative_slope,
    const int numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = input[idx] * static_cast<scalar_t>(multiplier);
        output[idx] = val > 0 ? val : val * static_cast<scalar_t>(negative_slope);
    }
}

torch::Tensor fused_mul_leakyrelu_cuda(
    torch::Tensor input,
    float multiplier,
    float negative_slope) {
    
    auto output = torch::empty_like(input);
    const int numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_mul_leakyrelu_cuda", ([&] {
        fused_mul_leakyrelu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            multiplier,
            negative_slope,
            numel
        );
    }));
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_mul_leakyrelu_cuda(
    torch::Tensor input,
    float multiplier,
    float negative_slope);
"""

fused_module = load_inline(
    name='fused_mul_leakyrelu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_mul_leakyrelu_cuda'],
    verbose=True
)

class Model(nn.Module):
    """
    Optimized model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    Uses a fused CUDA kernel for the multiplication and LeakyReLU operations.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.linear(x)
        return fused_module.fused_mul_leakyrelu_cuda(x, self.multiplier, self.negative_slope)

# Test parameters
batch_size = 128
in_features = 1024
out_features = 512
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
