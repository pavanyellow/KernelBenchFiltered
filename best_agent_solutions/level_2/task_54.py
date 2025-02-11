# level 2 index 54 agent name: KernelAgent 4o speedup: 1.26x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_kernels = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void OptimizedFusedKernel(
    const float* __restrict__ input, 
    const float* __restrict__ multiplier, 
    float* output,
    int channel_stride, int spatial_size, float negative_slope, 
    int num_elements) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int channel_index = (index / spatial_size) % channel_stride;
        
        float value = input[index] * multiplier[channel_index];
        
        // Directly applying GELU to Leaky ReLU output
        output[index] = gelu_approx(value > 0.0f ? value : negative_slope * value);
    }
}

void OptimizedFusedOperation(
    torch::Tensor input, torch::Tensor multiplier, torch::Tensor output, 
    int num_channels, int spatial_size, float negative_slope, int batch_size) {
    
    const int threads_per_block = 256;
    int num_elements = batch_size * num_channels * spatial_size;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    
    OptimizedFusedKernel<<<blocks_per_grid, threads_per_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), 
        multiplier.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_channels, spatial_size, negative_slope, num_elements);
}
"""

cpp_function_declaration = """
void OptimizedFusedOperation(
    torch::Tensor input, torch::Tensor multiplier, torch::Tensor output, 
    int num_channels, int spatial_size, float negative_slope, int batch_size);
"""

kernel_module = load_inline(
    name="optimized_fused_elementwise_kernel",
    cpp_sources=cpp_function_declaration,
    cuda_sources=cuda_kernels,
    functions=["OptimizedFusedOperation"],
    verbose=False
)

class OptimizedModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(OptimizedModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu_negative_slope = nn.LeakyReLU().negative_slope

    def forward(self, x):
        x = self.conv(x)

        # Allocate output tensor
        batch_size, num_channels, height, width = x.shape
        output = torch.empty_like(x, memory_format=torch.contiguous_format)

        # Launch the improved optimized kernel
        kernel_module.OptimizedFusedOperation(
            x, self.multiplier, output, num_channels,
            height * width, self.leaky_relu_negative_slope, batch_size
        )

        return output

Model = OptimizedModel
