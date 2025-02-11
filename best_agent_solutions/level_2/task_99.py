# level 2 index 99 agent name: KernelAgent 4o speedup: 1.85x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel
fused_linear_gelu_softmax_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu(float x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

__global__ void fused_linear_gelu_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_features,
    int out_features
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* shm_input = shared_data;
    float* shm_weights = &shared_data[in_features];
    float* shm_out = &shared_data[in_features + in_features * out_features];

    if(tid < in_features) {
        shm_input[tid] = input[row * in_features + tid];
    }
    for (int i = tid; i < out_features * in_features; i += blockDim.x) {
        shm_weights[i] = weight[i];
    }
    __syncthreads();

    if (tid < out_features) {
        float value = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            value += shm_input[i] * shm_weights[tid * in_features + i];
        }
        value += bias[tid];
        value = gelu(value);

        shm_out[tid] = value;
    }
    __syncthreads();

    if (tid < out_features) {
        float max_val = -1e20f;
        for (int i = 0; i < out_features; ++i) {
            max_val = max(max_val, shm_out[i]);
        }
        __syncthreads();

        float sum_exp = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            shm_out[i] = expf(shm_out[i] - max_val);
            sum_exp += shm_out[i];
        }
        __syncthreads();

        output[row * out_features + tid] = shm_out[tid] / sum_exp;
    }
}

void fused_linear_gelu_softmax(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_memory_size = (in_features + in_features * out_features + out_features) * sizeof(float);

    fused_linear_gelu_softmax_kernel<<<blocks, threads, shared_memory_size>>>(
        input.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        output.contiguous().data_ptr<float>(),
        in_features,
        out_features
    );
}
"""

fused_linear_gelu_softmax_cpp_source = """
void fused_linear_gelu_softmax(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output
);
"""

# Compile the inline CUDA code
fused_linear_gelu_softmax_module = load_inline(
    name='fused_linear_gelu_softmax',
    cpp_sources=fused_linear_gelu_softmax_cpp_source,
    cuda_sources=fused_linear_gelu_softmax_source,
    functions=['fused_linear_gelu_softmax'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _ = x.size()
        output = torch.empty((batch_size, self.weight.size(0)), device=x.device, dtype=x.dtype)
        fused_linear_gelu_softmax_module.fused_linear_gelu_softmax(x, self.weight, self.bias, output)
        return output
