# level 2 index 95 agent name: KernelAgent Claude 3.5 Sonnet speedup: 2.00x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for fused activation functions
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_impl(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_activations_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // Swish
        x = x * (1.0f / (1.0f + expf(-x)));
        
        // Tanh
        x = tanhf(x);
        
        // GELU
        x = gelu_impl(x);
        
        // Hardtanh removed as it's redundant after tanh
        
        output[idx] = x;
    }
}

torch::Tensor fused_activations_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    fused_activations_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_activations_cuda(torch::Tensor input);
"""

# Compile the CUDA kernel
fused_activation_module = load_inline(
    name='fused_activations',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_activations_cuda'],
    verbose=True
)

class Model(nn.Module):
    """
    Optimized version using fused CUDA kernel for activations with redundant hardtanh removed
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        # Initialize add_value the same way as original for consistency
        add_value = torch.randn(add_value_shape)
        # Combine the add_value into the bias
        self.matmul.bias.data.add_(add_value)

    def forward(self, x):
        x = self.matmul(x)  # Using PyTorch's optimized matmul
        # Apply all activations in one fused CUDA kernel, minus the redundant hardtanh
        x = fused_activation_module.fused_activations_cuda(x)
        return x
