# level 2 index 65 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.42x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float4 sigmoid4(float4 v) {
    float4 result;
    result.x = sigmoid(v.x);
    result.y = sigmoid(v.y);
    result.z = sigmoid(v.z);
    result.w = sigmoid(v.w);
    return result;
}

__device__ __forceinline__ float sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
}

__global__ void sigmoid_and_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const float pool_scale
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int elements_per_batch = channels * height * width;
    const float* input_batch = input + b * elements_per_batch;
    
    // Convert input pointer to float4 for vectorized loads
    const float4* input_vec4 = reinterpret_cast<const float4*>(input_batch);
    const int vec4_elements = elements_per_batch / 4;
    
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time using vectorized loads
    #pragma unroll 4
    for (int i = tid; i < vec4_elements; i += blockDim.x) {
        float4 in4 = input_vec4[i];
        float4 sig4 = sigmoid4(in4);
        thread_sum += sum4(sig4);
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_elements * 4;
    for (int i = remaining_start + tid; i < elements_per_batch; i += blockDim.x) {
        thread_sum += sigmoid(input_batch[i]);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Reduction in shared memory
    #pragma unroll
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) {
        output[b] = sdata[0] * pool_scale;
    }
}

torch::Tensor sigmoid_and_sum_cuda(
    torch::Tensor input,
    float pool_scale
) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];

    auto output = torch::empty({batch_size}, input.options());

    // Use fewer threads per block for better occupancy
    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem = threads * sizeof(float);

    sigmoid_and_sum_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        pool_scale
    );

    return output;
}
"""

cpp_source = """
torch::Tensor sigmoid_and_sum_cuda(torch::Tensor input, float pool_scale);
"""

module = load_inline(
    name='custom_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['sigmoid_and_sum_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_scale = 1.0 / (pool_kernel_size * pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = module.sigmoid_and_sum_cuda(x, self.pool_scale)
        return x
