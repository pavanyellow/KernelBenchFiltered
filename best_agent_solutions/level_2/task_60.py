# level 2 index 60 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.30x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fast sigmoid approximation
__device__ __forceinline__ float fast_sigmoid(float x) {
    return __fdividef(1.0f, (1.0f + __expf(-x)));
}

// Vectorized Swish kernel with better memory coalescing
template<int VEC_SIZE=4>
__global__ void fused_swish_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < num_elements; i += stride) {
        float4 in = __ldg(&input[i]);
        float4 out;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = reinterpret_cast<float*>(&in)[j];
            reinterpret_cast<float*>(&out)[j] = val * fast_sigmoid(val);
        }
        
        output[i] = out;
    }
}

// Improved GroupNorm + HardSwish kernel
template<int BLOCK_SIZE=256>
__global__ void fused_groupnorm_hardswish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const int N,
    const int C,
    const int DHW,
    const int groups,
    const float eps
) {
    extern __shared__ float shmem[];
    float* s_mean = shmem;
    float* s_var = &shmem[BLOCK_SIZE];
    
    const int group_size = (C / groups) * DHW;
    const int group_id = blockIdx.x / N;
    const int batch_id = blockIdx.x % N;
    const int thread_id = threadIdx.x;
    
    // Calculate group statistics
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    const int start_idx = batch_id * C * DHW + group_id * group_size;
    
    // Vectorized load when possible
    const int vec_elements = group_size / 4 * 4;
    const float4* input4 = reinterpret_cast<const float4*>(&input[start_idx]);
    
    #pragma unroll 4
    for (int i = thread_id; i < vec_elements/4; i += BLOCK_SIZE) {
        float4 vals = __ldg(&input4[i]);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = reinterpret_cast<float*>(&vals)[j];
            local_sum += val;
            local_sq_sum += val * val;
        }
    }
    
    // Handle remaining elements
    for (int i = vec_elements + thread_id; i < group_size; i += BLOCK_SIZE) {
        float val = __ldg(&input[start_idx + i]);
        local_sum += val;
        local_sq_sum += val * val;
    }
    
    // Parallel reduction
    s_mean[thread_id] = local_sum;
    s_var[thread_id] = local_sq_sum;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            s_mean[thread_id] += s_mean[thread_id + stride];
            s_var[thread_id] += s_var[thread_id + stride];
        }
        __syncthreads();
    }
    
    // Compute statistics
    const float rcp_group_size = __fdividef(1.0f, group_size);
    float mean = s_mean[0] * rcp_group_size;
    float variance = fmaxf(s_var[0] * rcp_group_size - mean * mean, 0.0f);
    float inv_std = rsqrtf(variance + eps);
    
    // Apply normalization and activation
    const int channels_per_group = C / groups;
    float4* output4 = reinterpret_cast<float4*>(&output[start_idx]);
    
    #pragma unroll 4
    for (int i = thread_id; i < vec_elements/4; i += BLOCK_SIZE) {
        float4 in4 = __ldg(&input4[i]);
        float4 out4;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = (i * 4 + j) / DHW + group_id * channels_per_group;
            float val = reinterpret_cast<float*>(&in4)[j];
            
            // Normalize
            val = (val - mean) * inv_std;
            val = val * __ldg(&gamma[c]) + __ldg(&beta[c]);
            
            // HardSwish
            float relu6 = min(max(val + 3.0f, 0.0f), 6.0f);
            reinterpret_cast<float*>(&out4)[j] = val * relu6 * 0.16666667f;
        }
        
        output4[i] = out4;
    }
    
    // Handle remaining elements
    for (int i = vec_elements + thread_id; i < group_size; i += BLOCK_SIZE) {
        int c = i / DHW + group_id * channels_per_group;
        float val = __ldg(&input[start_idx + i]);
        
        val = (val - mean) * inv_std;
        val = val * __ldg(&gamma[c]) + __ldg(&beta[c]);
        
        float relu6 = min(max(val + 3.0f, 0.0f), 6.0f);
        output[start_idx + i] = val * relu6 * 0.16666667f;
    }
}

torch::Tensor fused_swish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int numel = input.numel() / 4;
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    fused_swish_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        numel
    );
    
    return output;
}

torch::Tensor fused_groupnorm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int DHW = D * H * W;
    
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = N * groups;
    const int shared_mem = threads * 2 * sizeof(float);
    
    fused_groupnorm_hardswish_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, DHW,
        groups, eps
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_swish_cuda(torch::Tensor input);
torch::Tensor fused_groupnorm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
);
"""

activation_module = load_inline(
    name='activation_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_swish_cuda', 'fused_groupnorm_hardswish_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.groups = groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = activation_module.fused_swish_cuda(x)
        x = activation_module.fused_groupnorm_hardswish_cuda(
            x, self.gamma, self.beta, self.groups, self.eps
        )
        return x
