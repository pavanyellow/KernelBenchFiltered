# level 2 index 31 agent name: KernelAgent 4o speedup: 2.04x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel code for processing
cuda_source = """
#include <torch/extension.h>

extern "C" __global__
void min_plus_bias_scale(const float* __restrict__ input, 
                         float* __restrict__ output, 
                         const float* __restrict__ bias, 
                         const float constant_value, 
                         const float scaling_factor,
                         const int height, const int width, 
                         const int channels, const int batch_size) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (h < height && w < width && c < channels) {
        int idx = n * channels * height * width + c * height * width + h * width + w;
        float input_val = input[idx];
        float min_val = min(input_val, constant_value);
        float biased_val = min_val + bias[c];
        float scaled_val = biased_val * scaling_factor;
        output[idx] = scaled_val;
    }
}

torch::Tensor min_plus_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float constant_value, float scaling_factor) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const dim3 threads_per_block(width, height);
    const dim3 num_blocks(batch_size, channels);

    min_plus_bias_scale<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_value,
        scaling_factor,
        height,
        width,
        channels,
        batch_size);

    return output;
}
"""

cpp_source = """
torch::Tensor min_plus_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float constant_value, float scaling_factor);
"""

min_plus_bias_scale_module = load_inline(
    name='min_plus_bias_scale',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['min_plus_bias_scale_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = min_plus_bias_scale_module.min_plus_bias_scale_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        return x
