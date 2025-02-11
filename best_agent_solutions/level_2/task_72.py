# level 2 index 72 agent name: KernelAgent 4o speedup: 2.69x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA kernel for enhanced Avg Pool and Batch Norm operation
cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void optimized_avg_pool3d_batch_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int channels, int depth, int height, int width,
    float epsilon) {

    extern __shared__ float shared_mem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * (depth / 4) * (height / 4) * (width / 4)) return;

    int out_depth = depth / 4;
    int out_height = height / 4;
    int out_width = width / 4;
    
    int b = idx / (channels * out_depth * out_height * out_width);
    int c = (idx / (out_depth * out_height * out_width)) % channels;
    int d = (idx / (out_height * out_width)) % out_depth;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    int in_d = d * 4;
    int in_h = h * 4;
    int in_w = w * 4;

    float sum = 0.0f;

    #pragma unroll 4
    for (int fd = 0; fd < 4; ++fd) {
        for (int fh = 0; fh < 4; ++fh) {
            for (int fw = 0; fw < 4; ++fw) {
                int in_idx = 
                    (((b * channels + c) * depth + (in_d + fd)) * height + (in_h + fh)) * width + (in_w + fw);
                sum += input[in_idx];
            }
        }
    }

    float avg_val = sum / 64.0f;  // 4x4x4 window pooling
    float inv_std = rsqrtf(var[c] + epsilon);

    output[idx] = (avg_val - mean[c]) * inv_std * weight[c] + bias[c];
}

torch::Tensor optimized_avg_pool3d_batch_norm_cuda(
    torch::Tensor input, 
    torch::Tensor mean, 
    torch::Tensor var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float epsilon) {

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto options = torch::TensorOptions(input.device()).dtype(input.dtype());

    auto output = torch::empty({batch_size, channels, depth / 4, height / 4, width / 4}, options);

    const int output_size = batch_size * channels * (depth / 4) * (height / 4) * (width / 4);
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;

    optimized_avg_pool3d_batch_norm_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        mean.data_ptr<float>(), 
        var.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        batch_size, 
        channels, 
        depth, 
        height, 
        width, 
        epsilon
    );

    return output;
}
"""

cpp_source = """
torch::Tensor optimized_avg_pool3d_batch_norm_cuda(
    torch::Tensor input, 
    torch::Tensor mean, 
    torch::Tensor var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float epsilon);
"""

# Compile the CUDA code
module = load_inline(
    name="optimized_kernels_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_sources,
    functions=["optimized_avg_pool3d_batch_norm_cuda"],
    verbose=True,
)

class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias_shape=None):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.epsilon = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        
        if self.training:
            mean = x.mean([0, 2, 3, 4])
            var = x.var([0, 2, 3, 4], unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var
        
        x = module.optimized_avg_pool3d_batch_norm_cuda(x, mean, var, self.bn_weight, self.bn_bias, self.epsilon)
        return x
