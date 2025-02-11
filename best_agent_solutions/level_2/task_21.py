# level 2 index 21 agent name: KernelAgent 4o speedup: 1.23x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for further fused operations
kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void enhanced_fuse_bias_scale_sigmoid(const float* __restrict__ input, 
                                                 const float* __restrict__ bias, 
                                                 const float* __restrict__ scale, 
                                                 float* __restrict__ output, 
                                                 int c, int hw) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int idx = threadIdx.x;
    
    if (channel_idx < c && batch_idx < gridDim.y) {
        float b = bias[channel_idx];
        float s = scale[channel_idx];
        int base_idx = (batch_idx * c + channel_idx) * hw;
      
        for (int i = idx; i < hw; i += blockDim.x) {
            float val = input[base_idx + i];
            val = (val + b) * s;
            output[base_idx + i] = sigmoid(val);
        }
    }
}

torch::Tensor enhanced_fuse_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    int n = input.size(0);
    int c = input.size(1);
    int h = input.size(2);
    int w = input.size(3);
    auto output = torch::empty_like(input);

    int hw = h * w;
    dim3 block(256);
    dim3 grid(c, n);

    enhanced_fuse_bias_scale_sigmoid<<<grid, block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        c, hw
    );

    return output;
}
"""

cpp_source = "torch::Tensor enhanced_fuse_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);"

# Compile the inline CUDA code for the further optimized fused operation
optim_native_module = load_inline(
    name='enhanced_fuse_bias_scale_sigmoid',
    cpp_sources=cpp_source,
    cuda_sources=kernel_source,
    functions=['enhanced_fuse_bias_scale_sigmoid_cuda'],
    verbose=False
)

class Model(nn.Module):
    """
    Highly Optimized Model that performs a convolution, further fused bias addition, 
    scaling, sigmoid activation, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = optim_native_module.enhanced_fuse_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
