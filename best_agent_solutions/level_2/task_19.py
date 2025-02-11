# level 2 index 19 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.34x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_fast(float x) {
    return x * 0.5f * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C" __global__ void fused_gelu_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_channels,
    const int height,
    const int width,
    const int num_groups,
    const int channels_per_group,
    const float eps
) {
    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_variance = &shared_mem[32];
    
    const int tid = threadIdx.x;
    const int group_idx = blockIdx.x % num_groups;
    const int batch_idx = blockIdx.x / num_groups;
    const int hw = height * width;
    const int chw = num_channels * hw;
    
    if (batch_idx >= batch_size) return;
    
    const int start_channel = group_idx * channels_per_group;
    const int end_channel = start_channel + channels_per_group;
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int c = start_channel; c < end_channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = tid; w < width; w += blockDim.x) {
                const int idx = batch_idx * chw + c * hw + h * width + w;
                float val = gelu_fast(input[idx]);
                sum += val;
                sq_sum += val * val;
            }
        }
    }
    
    sum = warp_reduce_sum(sum);
    sq_sum = warp_reduce_sum(sq_sum);
    
    if (tid < 32) {
        s_mean[tid] = sum;
        s_variance[tid] = sq_sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        float final_sum = 0.0f;
        float final_sq_sum = 0.0f;
        
        for (int i = 0; i < 32; i++) {
            final_sum += s_mean[i];
            final_sq_sum += s_variance[i];
        }
        
        const float size = channels_per_group * hw;
        const float mean = final_sum / size;
        const float variance = (final_sq_sum / size) - (mean * mean);
        
        s_mean[0] = mean;
        s_variance[0] = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    const float mean = s_mean[0];
    const float inv_std = s_variance[0];
    
    for (int c = start_channel; c < end_channel; c++) {
        const float w = weight[c];
        const float b = bias[c];
        
        for (int h = 0; h < height; h++) {
            for (int w_idx = tid; w_idx < width; w_idx += blockDim.x) {
                const int idx = batch_idx * chw + c * hw + h * width + w_idx;
                float val = gelu_fast(input[idx]);
                output[idx] = (val - mean) * inv_std * w + b;
            }
        }
    }
}

torch::Tensor fused_gelu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int channels_per_group = num_channels / num_groups;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size * num_groups;
    const int shared_mem_size = 64 * sizeof(float);
    
    fused_gelu_groupnorm_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        channels_per_group,
        eps
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_gelu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

fused_module = load_inline(
    name='fused_gelu_groupnorm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_gelu_groupnorm_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5
        
        # Initialize parameters exactly like GroupNorm
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_module.fused_gelu_groupnorm_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
