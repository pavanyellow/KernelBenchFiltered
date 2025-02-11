# level 2 index 98 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.96x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float fast_gelu(float x) {
    // Faster GELU approximation
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + coeff * x3)));
}

struct Float4 {
    float4 data;
    
    __device__ __forceinline__ Float4() {}
    
    __device__ __forceinline__ Float4(float4 val) : data(val) {}
    
    __device__ __forceinline__ Float4(float val) {
        data.x = val;
        data.y = val;
        data.z = val;
        data.w = val;
    }
    
    __device__ __forceinline__ float max_val() {
        return fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w));
    }
};

__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__global__ void fused_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const float scale_factor
) {
    extern __shared__ float smem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    
    if (bid >= batch_size) return;
    
    // Use vectorized loads when possible
    const int vec_features = features / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input + bid * features);
    float thread_max = -INFINITY;
    
    // Process 4 elements at a time using float4
    #pragma unroll 4
    for (int i = tid; i < vec_features; i += blockDim.x) {
        float4 vec = input_vec[i];
        Float4 vals(vec);
        
        // Process each element
        float4 results;
        results.x = fast_gelu(vals.data.x) * scale_factor;
        results.y = fast_gelu(vals.data.y) * scale_factor;
        results.z = fast_gelu(vals.data.z) * scale_factor;
        results.w = fast_gelu(vals.data.w) * scale_factor;
        
        Float4 res(results);
        thread_max = fmaxf(thread_max, res.max_val());
    }
    
    // Handle remaining elements
    for (int i = vec_features * 4 + tid; i < features; i += blockDim.x) {
        float val = input[bid * features + i];
        float gelu_val = fast_gelu(val);
        float scaled = gelu_val * scale_factor;
        thread_max = fmaxf(thread_max, scaled);
    }
    
    // Warp reduction
    thread_max = warp_reduce_max(thread_max);
    
    // Block reduction using shared memory
    if (lane == 0) {
        smem[wid] = thread_max;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        float block_max = -INFINITY;
        for (int i = 0; i < blockDim.x / 32; ++i) {
            block_max = fmaxf(block_max, smem[i]);
        }
        output[bid] = block_max;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float scale_factor) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    
    auto output = torch::empty({batch_size}, 
                             torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    const int threads = 256;
    const int blocks = batch_size;
    const int smem_size = (threads / 32) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_op_cuda", ([&] {
        fused_gelu_scale_max_kernel<<<blocks, threads, smem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            features,
            scale_factor
        );
    }));
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, float scale_factor);
"""

fused_module = load_inline(
    name='fused_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        # Pre-pool the weights and bias
        self.pool_kernel_size = pool_kernel_size
        temp_linear = nn.Linear(in_features, out_features)
        
        # Reshape weights to do pooling
        weight = temp_linear.weight.view(-1, pool_kernel_size, in_features)
        weight = weight.mean(dim=1)
        bias = temp_linear.bias.view(-1, pool_kernel_size).mean(dim=1)
        
        # Create new linear layer with pooled weights
        self.matmul = nn.Linear(in_features, out_features // pool_kernel_size)
        self.matmul.weight.data = weight
        self.matmul.bias.data = bias
        
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul(x)
        return fused_module.fused_op_cuda(x, self.scale_factor)
