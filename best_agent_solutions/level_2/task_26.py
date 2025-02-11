# level 2 index 26 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.48x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel with optimized memory access and shared memory
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * min(max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ add_tensor,
    float* __restrict__ output,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W
) {
    // Use 2D grid for better occupancy
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int channel_idx = blockIdx.y;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int spatial_size = D * H * W;
    if (spatial_idx >= spatial_size) return;
    
    // Calculate 3D position
    const int d = spatial_idx / (H * W);
    const int hw = spatial_idx % (H * W);
    const int h = hw / W;
    const int w = hw % W;
    
    // Calculate global memory index
    const int idx = batch_idx * (C * D * H * W) + 
                   channel_idx * (D * H * W) + 
                   d * (H * W) + 
                   h * W + 
                   w;
    
    // Coalesced memory access
    const float in_val = input[idx];
    const float add_val = add_tensor[idx];
    
    // Compute and write result
    output[idx] = hardswish(in_val + add_val);
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor input, torch::Tensor add_tensor) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int threads = 256;
    const int spatial_size = depth * height * width;
    const int spatial_blocks = (spatial_size + threads - 1) / threads;
    
    dim3 blocks(spatial_blocks, channels, batch_size);
    dim3 threads_per_block(threads);
    
    fused_add_hardswish_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        add_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_add_hardswish_cuda(torch::Tensor input, torch::Tensor add_tensor);
"""

fused_module = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_add_hardswish_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution.
        Returns:
            torch.Tensor: Output tensor after HardSwish activation.
        """
        x = self.conv_transpose(x)
        return fused_module.fused_add_hardswish_cuda(x, add_input)

# Keep the same initialization parameters
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [
        torch.randn(batch_size, in_channels, D, H, W),
        torch.randn(batch_size, out_channels, D*stride, H*stride, W*stride)
    ]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
