# level 2 index 50 agent name: KernelAgent 4o speedup: 1.30x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized fused operations
fused_cuda_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_operations_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                        const float* __restrict__ bias, float scale1, float scale2,
                                        int N, int C, int D, int H, int W,
                                        int pooled_D, int pooled_H, int pooled_W) {
    // Allocate shared memory for bias
    extern __shared__ float shared_bias[];

    int threadId = threadIdx.x;
    
    // Load bias into shared memory
    if (threadId < C) {
        shared_bias[threadId] = bias[threadId];
    }
    __syncthreads();

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadId;
   
    // Proceed if the global index is within the bounds
    if (idx < N * C * pooled_D * pooled_H * pooled_W) {
        // Calculate pooled indices
        int pw = idx % pooled_W;
        int ph = (idx / pooled_W) % pooled_H;
        int pd = (idx / (pooled_W * pooled_H)) % pooled_D;
        int c = (idx / (pooled_W * pooled_H * pooled_D)) % C;
        int n = idx / (pooled_W * pooled_H * pooled_D * C);

        // Compute starting input indices based on pooling
        int start_w = pw * 2;
        int start_h = ph * 2;
        int start_d = pd * 2;

        float pooled_value = 0.0;

        // Perform average pooling over 2x2x2 region with scaling
        for (int z = 0; z < 2; ++z) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    int input_idx = ((n * C + c) * D + (start_d + z)) * H * W + (start_h + y) * W + (start_w + x);
                    pooled_value += input[input_idx] * scale1 / 8.0f;
                }
            }
        }

        // Add bias and apply second scale
        float result = (pooled_value + shared_bias[c]) * scale2;

        // Write to output
        int output_idx = (((n * C) + c) * pooled_D + pd) * pooled_H * pooled_W + ph * pooled_W + pw;
        output[output_idx] = result;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bias, float scale1, float scale2) {
    // Determine the shape of avg pooling output
    int pooled_D = input.size(2) / 2;
    int pooled_H = input.size(3) / 2;
    int pooled_W = input.size(4) / 2;
    auto output = torch::empty({input.size(0), input.size(1), pooled_D, pooled_H, pooled_W}, input.options());

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    const int block_size = 256;
    const int num_blocks = (N * C * pooled_D * pooled_H * pooled_W + block_size - 1) / block_size;
    const int shared_memory_size = C * sizeof(float);  // Allocate shared memory for bias

    fused_operations_kernel<<<num_blocks, block_size, shared_memory_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(), scale1, scale2,
        N, C, D, H, W, pooled_D, pooled_H, pooled_W
    );

    return output;
}
"""

# Define the CPP source for the CUDA extension
fused_cuda_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bias, float scale1, float scale2);
"""

# Compile the CUDA extension
fused_module = load_inline(
    name='fused_ops',
    cpp_sources=fused_cuda_cpp_source,
    cuda_sources=fused_cuda_kernel,
    functions=['fused_forward_cuda'],
    verbose=True
)

# Define the PyTorch Model that uses the optimized fused CUDA kernel
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = scale1
        self.scale2 = scale2
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_module.fused_forward_cuda(x, self.bias, self.scale1, self.scale2)
        return x
