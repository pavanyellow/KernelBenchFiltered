# level 2 index 73 agent name: KernelAgent 4o speedup: 1.13x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized CUDA kernel for fused Batch Normalization and scaling
fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ scaled_weight, 
    const float* __restrict__ scaled_bias, 
    float* __restrict__ output, 
    int num_channels, 
    int spatial_size) 
{
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (spatial_idx < spatial_size) {
        float inv_var = rsqrtf(var[channel] + 1e-5f);
        int idx = batch * (num_channels * spatial_size) + channel * spatial_size + spatial_idx;
        output[idx] = ((input[idx] - mean[channel]) * inv_var * scaled_weight[channel]) + scaled_bias[channel];
    }
}

torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float scaling_factor)
{
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);
    
    auto output = torch::empty_like(input, input.options());
    auto scaled_weight = weight * scaling_factor;
    auto scaled_bias = bias * scaling_factor;

    dim3 threads_per_block(256);
    dim3 num_blocks(batch_size, num_channels, (spatial_size + threads_per_block.x - 1) / threads_per_block.x);

    fused_bn_scale_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        scaled_weight.data_ptr<float>(),
        scaled_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_channels,
        spatial_size
    );

    return output;
}
"""

fused_bn_scale_cpp_source = """
torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float scaling_factor);
"""

# Compile the CUDA module for fused operations
fused_bn_scale_native_module = load_inline(
    name='fused_bn_scale',
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=['fused_bn_scale_cuda'],
    verbose=False
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Consider making conv layer computation async
        x = self.conv(x)
        
        if self.training:
            # Use PyTorch's fused implementation for performance
            x = F.batch_norm(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, training=True)
            x *= self.scaling_factor
        else:
            # Use custom kernel for performance in eval mode
            x = fused_bn_scale_native_module.fused_bn_scale_cuda(
                x, 
                self.bn.running_mean, 
                self.bn.running_var, 
                self.bn.weight, 
                self.bn.bias, 
                self.scaling_factor
            )
        return x

# Example usage
if __name__ == "__main__":
    model = Model(in_channels=3, out_channels=16, kernel_size=3, scaling_factor=2.0).cuda()
    x = torch.randn(128, 3, 32, 32, device='cuda')
    model.eval()
    output = model(x)
    print(output.shape)
