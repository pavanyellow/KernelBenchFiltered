# level 2 index 52 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.17x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_softplus(float x) {
    const float threshold = 20.0f;
    if (x > threshold) return x;
    if (x < -threshold) return __expf(x);
    return __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float fast_tanh(float x) {
    float ex = __expf(2.0f * x);
    return (ex - 1.0f) / (ex + 1.0f);
}

// Training mode kernel with 2D block structure
__global__ void fused_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int h_start = by * blockDim.y + ty;
    const int w_start = bx * blockDim.x + tx;
    
    if (h_start < height && w_start < width) {
        const int idx = bz * channels * height * width + 
                       h_start * width + w_start;
        
        float val = input[idx];
        float sp = fast_softplus(val);
        float th = fast_tanh(sp);
        output[idx] = th * val;
    }
}

// Inference mode kernel with shared memory for BatchNorm parameters
__global__ void fused_activation_batchnorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const float eps
) {
    extern __shared__ float shared_params[];
    float* s_mean = &shared_params[0];
    float* s_var = &shared_params[channels];
    float* s_weight = &shared_params[2 * channels];
    float* s_bias = &shared_params[3 * channels];

    // Load channel parameters into shared memory
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    
    for (int i = tid; i < channels; i += block_size) {
        s_mean[i] = running_mean[i];
        s_var[i] = running_var[i];
        s_weight[i] = weight[i];
        s_bias[i] = bias[i];
    }
    __syncthreads();

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int h_start = by * blockDim.y + ty;
    const int w_start = bx * blockDim.x + tx;
    
    if (h_start < height && w_start < width) {
        for (int c = 0; c < channels; c++) {
            const int idx = bz * channels * height * width +
                          c * height * width +
                          h_start * width + w_start;
            
            float val = input[idx];
            float sp = fast_softplus(val);
            float th = fast_tanh(sp);
            val = th * val;

            // Apply BatchNorm using shared memory
            output[idx] = s_weight[c] * (val - s_mean[c]) / 
                         sqrtf(s_var[c] + eps) + s_bias[c];
        }
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size * channels
    );
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}

torch::Tensor fused_activation_batchnorm_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size
    );
    
    const int shared_mem_size = 4 * channels * sizeof(float);
    
    fused_activation_batchnorm_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        eps
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor input);
torch::Tensor fused_activation_batchnorm_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

fused_module = load_inline(
    name='fused_operations',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_activation_cuda', 'fused_activation_batchnorm_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv(x)
        
        if self.training:
            x = fused_module.fused_activation_cuda(x)
            x = self.bn(x)
        else:
            x = fused_module.fused_activation_batchnorm_cuda(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.eps
            )
        
        return x

def get_inputs():
    return [torch.randn(128, 3, 32, 32)]

def get_init_inputs():
    return [3, 16, 3]
