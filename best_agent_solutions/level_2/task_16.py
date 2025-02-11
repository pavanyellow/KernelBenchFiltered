# level 2 index 16 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.65x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float fast_mish(float x) {
    // Fast approximate mish using fast exp and tanh
    float e = __expf(x);
    float n = e + 1.0f;
    float e2 = e * e;
    float n2 = n * n;
    float tanhf = (e2 - 1.0f) / (e2 + 1.0f);
    return x * tanhf;
}

__device__ __forceinline__ float4 fast_mish_vector(float4 x) {
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* val = (float*)&x + i;
        float* res = (float*)&result + i;
        *res = fast_mish(*val);
    }
    return result;
}

__device__ __forceinline__ float4 hardtanh_scale_vector(float4 x, float add_value, float scale) {
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* val = (float*)&x + i;
        float tmp = __fadd_rn(*val, add_value);
        tmp = __fminf(1.0f, __fmaxf(-1.0f, tmp));
        float* res = (float*)&result + i;
        *res = __fmul_rn(tmp, scale);
    }
    return result;
}

__global__ void fused_post_conv_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float add_value,
    const float scale,
    const int N,
    const int C,
    const int H,
    const int W
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int h = (blockIdx.x * block_size + tid) / W;
    const int w = (blockIdx.x * block_size + tid) % W;
    
    if (h >= H || w >= W) return;
    
    const int input_idx = ((n * C + c) * H + h) * W + w;
    const int output_idx = input_idx;
    
    // Load data to shared memory
    shared_mem[tid] = input[input_idx];
    __syncthreads();
    
    // Process in vectors of 4 when possible
    if (tid < block_size/4) {
        float4 data;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float* ptr = (float*)&data + i;
            *ptr = shared_mem[tid*4 + i];
        }
        
        data = fast_mish_vector(data);
        data = hardtanh_scale_vector(data, add_value, scale);
        
        // Store results
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (tid*4 + i < block_size && h*W + w + i < H*W) {
                output[output_idx + i] = ((float*)&data)[i];
            }
        }
    }
}

torch::Tensor fused_post_conv_cuda(
    torch::Tensor input,
    float add_value,
    float scale
) {
    auto output = torch::empty_like(input);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    const int threads = 256;
    dim3 blocks(
        (H * W + threads - 1) / threads,
        C,
        N
    );
    
    const int shared_mem_size = threads * sizeof(float);
    
    fused_post_conv_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        add_value,
        scale,
        N, C, H, W
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_post_conv_cuda(
    torch::Tensor input,
    float add_value,
    float scale
);
"""

fused_ops_module = load_inline(
    name='fused_post_conv_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_post_conv_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        torch.backends.cudnn.benchmark = True
        # Ensure weights are in correct format for best performance
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous()
        
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # Ensure input is contiguous and in correct format
        x = x.contiguous()
        x = self.conv_transpose(x)
        return fused_ops_module.fused_post_conv_cuda(x, self.add_value, self.scale)

def get_inputs():
    return [torch.randn(128, 32, 16, 16)]

def get_init_inputs():
    return [32, 64, 4, 2, 1, 1, 0.5, 2]
