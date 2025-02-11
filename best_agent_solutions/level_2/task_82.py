# level 2 index 82 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.64x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_tanh(float x) {
    float ex2 = __expf(2*x);
    return (ex2 - 1.0f) / (ex2 + 1.0f);
}

__global__ void fused_ops_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int output_height,
    const int output_width,
    const int pool_size
) {
    extern __shared__ float shared_mem[];
    float* shared_bias = shared_mem;
    float* shared_block = shared_mem + channels;
    
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();

    const int thread_work = 4;
    const int stride = gridDim.x * blockDim.x * thread_work;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * thread_work;
    const int total_size = batch_size * channels * output_height * output_width;

    #pragma unroll
    for (int work = 0; work < thread_work && idx < total_size; work++, idx++) {
        const int w_out = idx % output_width;
        const int h_out = (idx / output_width) % output_height;
        const int c = (idx / (output_height * output_width)) % channels;
        const int b = idx / (channels * output_height * output_width);

        const int h_in = h_out * pool_size;
        const int w_in = w_out * pool_size;
        
        const int in_base = ((b * channels + c) * height + h_in) * width + w_in;
        float max_val = -INFINITY;

        #pragma unroll
        for (int ph = 0; ph < pool_size; ph++) {
            const float* in_row = input + in_base + ph * width;
            #pragma unroll
            for (int pw = 0; pw < pool_size; pw++) {
                // Combined scaling_factor * tanh(x) + bias into single FMA
                float val = fmaf(scaling_factor, fast_tanh(in_row[pw]), shared_bias[c]);
                max_val = fmaxf(max_val, val);
            }
        }

        output[idx] = max_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    const auto output_height = height / pool_size;
    const auto output_width = width / pool_size;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int thread_work = 4;
    const int threads = 256;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = min(4096, (total_elements + (threads * thread_work) - 1) / (threads * thread_work));
    
    const int shared_mem_size = (channels + threads) * sizeof(float);
    
    fused_ops_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        height,
        width,
        output_height,
        output_width,
        pool_size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
);
"""

fused_ops_module = load_inline(
    name='fused_ops_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops_module.fused_ops_cuda(
            x,
            self.bias.view(-1),
            self.scaling_factor,
            self.pool_size
        )
        return x
