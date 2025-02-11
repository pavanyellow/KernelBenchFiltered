# level 2 index 1 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.24x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 16
#define SHARED_MEM_SIZE 64  // Maximum number of bias elements we expect

__global__ void fused_conv_relu_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W
) {
    __shared__ float shared_bias[SHARED_MEM_SIZE];
    
    // Load bias into shared memory
    if (threadIdx.x < C) {
        shared_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * H * W;
    const int elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const int block_start = blockIdx.x * elements_per_block;
    
    // Process multiple elements per thread using float4
    float4* output_vec4 = reinterpret_cast<float4*>(output);
    const int vec4_elements = total_elements / 4;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD/4; ++i) {
        const int vec4_idx = block_start/4 + threadIdx.x + i*blockDim.x;
        if (vec4_idx < vec4_elements) {
            float4 val = output_vec4[vec4_idx];
            
            // Calculate channel indices for each component
            const int base_idx = vec4_idx * 4;
            const int c0 = (base_idx / (H * W)) % C;
            const int c1 = ((base_idx + 1) / (H * W)) % C;
            const int c2 = ((base_idx + 2) / (H * W)) % C;
            const int c3 = ((base_idx + 3) / (H * W)) % C;
            
            // ReLU + bias
            val.x = fmaxf(val.x, 0.0f) + shared_bias[c0];
            val.y = fmaxf(val.y, 0.0f) + shared_bias[c1];
            val.z = fmaxf(val.z, 0.0f) + shared_bias[c2];
            val.w = fmaxf(val.w, 0.0f) + shared_bias[c3];
            
            output_vec4[vec4_idx] = val;
        }
    }
    
    // Handle remaining elements
    const int rem_start = (total_elements/4)*4;
    const int rem_tid = block_start + threadIdx.x;
    if (rem_tid < total_elements - rem_start) {
        const int idx = rem_start + rem_tid;
        const int c = (idx / (H * W)) % C;
        float val = output[idx];
        output[idx] = fmaxf(val, 0.0f) + shared_bias[c];
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_elements = N * C * H * W;
    
    const int threads = BLOCK_SIZE;
    const int elements_per_block = threads * ELEMENTS_PER_THREAD;
    const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    
    fused_conv_relu_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W
    );
    
    return input;
}
"""

cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);
"""

fused_module = load_inline(
    name='fused_ops_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        return fused_module.fused_ops_cuda(x, self.bias.view(-1))

def get_inputs():
    return [torch.randn(128, 3, 32, 32)]

def get_init_inputs():
    return [3, 16, 3, (16, 1, 1)]
