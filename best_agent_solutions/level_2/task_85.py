# level 2 index 85 agent name: KernelAgent 4o speedup: 1.24x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F

optimized_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE_WIDTH = 16;
__global__ void optimized_kernel(const float* __restrict__ input, const float* __restrict__ scale, 
                                 float* __restrict__ output, int batch_size, int num_channels, int h_in, int w_in,
                                 int h_out, int w_out, int pool_size, float clamp_min, float clamp_max) {
    
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int h_out_idx = by * TILE_WIDTH + ty;
    int w_out_idx = bx * TILE_WIDTH + tx;

    if (h_out_idx < h_out && w_out_idx < w_out) {
        int c_idx = blockIdx.z % num_channels;
        int b_idx = blockIdx.z / num_channels;

        float max_value = clamp_min;
        float current_scale = scale[c_idx];

        int h_start = h_out_idx * pool_size;
        int w_start = w_out_idx * pool_size;

        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int h_offset = h_start + ph;
                int w_offset = w_start + pw;

                if (h_offset < h_in && w_offset < w_in) {
                    int input_idx = ((b_idx * num_channels + c_idx) * h_in + h_offset) * w_in + w_offset;
                    float val = input[input_idx] * current_scale;
                    max_value = max(max_value, min(max(val, clamp_min), clamp_max));
                }
            }
        }
        output[(b_idx * num_channels * h_out + c_idx * h_out + h_out_idx) * w_out + w_out_idx] = max_value;
    }
}

torch::Tensor optimized_scale_clamp_pool(torch::Tensor input, torch::Tensor scale, int pool_size, float clamp_min, float clamp_max) {
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int h_in = input.size(2);
    int w_in = input.size(3);
    int h_out = h_in / pool_size;
    int w_out = w_in / pool_size;

    auto output = torch::empty({batch_size, num_channels, h_out, w_out}, input.options());

    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((w_out + TILE_WIDTH - 1) / TILE_WIDTH, (h_out + TILE_WIDTH - 1) / TILE_WIDTH, batch_size * num_channels);

    optimized_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scale.data_ptr<float>(), 
        output.data_ptr<float>(), batch_size, num_channels,
        h_in, w_in, h_out, w_out, pool_size, clamp_min, clamp_max
    );

    return output;
}
"""

cpp_source_optimized = "torch::Tensor optimized_scale_clamp_pool(torch::Tensor input, torch::Tensor scale, int pool_size, float clamp_min, float clamp_max);"

# Compile the optimized CUDA code
optimized_native_module = load_inline(
    name='optimized_scale_clamp_pool_v4',
    cpp_sources=cpp_source_optimized,
    cuda_sources=optimized_cuda_source,
    functions=['optimized_scale_clamp_pool'],
    verbose=True,
)

class Model(nn.Module):
    """
    Model that performs convolution, group normalization with embedded scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)

        # Use optimized CUDA kernel for scaling, clamping, and pooling
        x = optimized_native_module.optimized_scale_clamp_pool(x, self.scale, self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
        return x

# Helper functions for creating inputs
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
