# level 2 index 15 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.74x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
struct Float4 {
    scalar_t v[4];
    __device__ Float4() {}
    __device__ Float4(scalar_t val) {
        v[0] = v[1] = v[2] = v[3] = val;
    }
};

// Optimized stats kernel using float4 loads
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ means,
    scalar_t* __restrict__ vars,
    const int N, const int C, const int DHW
) {
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_sq_sum = &shared[blockDim.x];
    
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int c = blockIdx.x;
    
    // Use float4 for coalesced memory access
    const int vec_size = 4;
    const int vec_DHW = DHW / vec_size;
    const Float4<scalar_t>* vec_input = reinterpret_cast<const Float4<scalar_t>*>(
        &input[n * C * DHW + c * DHW]);
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < vec_DHW; i += blockDim.x) {
        Float4<scalar_t> vec = vec_input[i];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = vec.v[j];
            sum += val;
            sq_sum += val * val;
        }
    }
    
    // Handle remaining elements
    const int remain_start = vec_DHW * vec_size;
    for (int i = remain_start + tid; i < DHW; i += blockDim.x) {
        float val = input[n * C * DHW + c * DHW + i];
        sum += val;
        sq_sum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
    }
    
    // Block-level reduction
    if (tid % warpSize == 0) {
        shared_sum[tid/warpSize] = sum;
        shared_sq_sum[tid/warpSize] = sq_sum;
    }
    __syncthreads();
    
    if (tid < warpSize) {
        sum = (tid < blockDim.x/warpSize) ? shared_sum[tid] : 0;
        sq_sum = (tid < blockDim.x/warpSize) ? shared_sq_sum[tid] : 0;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
        }
        
        if (tid == 0) {
            const float mean = sum / DHW;
            means[n * C + c] = mean;
            vars[n * C + c] = (sq_sum / DHW) - (mean * mean);
        }
    }
}

// Optimized normalize kernel with fused mean subtraction
template <typename scalar_t>
__global__ void normalize_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ means,
    const scalar_t* __restrict__ vars,
    const scalar_t* __restrict__ weight,
    const int N, const int C, const int DHW,
    const float eps
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int c = blockIdx.y;
    const int n = blockIdx.z;
    
    const int stride = blockDim.x * gridDim.x;
    const int offset = n * C * DHW + c * DHW;
    const scalar_t mean = means[n * C + c];
    const scalar_t var = vars[n * C + c];
    const scalar_t scale = weight[c] * rsqrtf(var + eps);
    
    #pragma unroll 4
    for (int i = bid * blockDim.x + tid; i < DHW; i += stride) {
        const scalar_t val = input[offset + i];
        output[offset + i] = (val - mean) * scale;
    }
}

std::vector<torch::Tensor> fused_bn_mean_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int DHW = input.size(2) * input.size(3) * input.size(4);
    
    auto means = torch::empty({N, C}, input.options());
    auto vars = torch::empty({N, C}, input.options());
    auto output = torch::empty_like(input);
    
    // Stats kernel config
    const int threads_stats = 256;
    const dim3 blocks_stats(C, N);
    const int shared_mem_size = 2 * threads_stats * sizeof(float);
    
    // Normalize kernel config
    const int threads_norm = 256;
    const int blocks_x = (DHW + threads_norm - 1) / threads_norm;
    const dim3 blocks_norm(blocks_x, C, N);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_bn_mean_sub_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            means.data_ptr<scalar_t>(),
            vars.data_ptr<scalar_t>(),
            N, C, DHW
        );
        
        normalize_kernel<scalar_t><<<blocks_norm, threads_norm>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            means.data_ptr<scalar_t>(),
            vars.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            N, C, DHW,
            eps
        );
    }));
    
    return {output, means, vars};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_bn_mean_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
);
"""

fused_module = load_inline(
    name='fused_bn_mean_sub',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_bn_mean_sub_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, bias=bias)
        
        self.register_parameter('bn_weight', nn.Parameter(torch.ones(out_channels)))
        self.register_parameter('bn_bias', nn.Parameter(torch.zeros(out_channels)))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        
        if self.training:
            output, batch_mean, batch_var = fused_module.fused_bn_mean_sub_cuda(
                x, self.bn_weight, self.eps)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                  self.momentum * batch_mean.mean(0)
                self.running_var = (1 - self.momentum) * self.running_var + \
                                 self.momentum * batch_var.mean(0)
        else:
            output = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.bn_weight,
                None,
                training=False,
                momentum=self.momentum,
                eps=self.eps
            )
        
        return output
