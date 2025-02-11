# level 2 index 92 agent name: KernelAgent O3 Mini High speedup: 2.73x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#define DIVIDE_ROUND_UP(x, y) (((x) + (y) - 1) / (y))

// --------------------------------------------------------------------
// Warp-level reduction for summing float values within a warp.
__forceinline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// --------------------------------------------------------------------
// Kernel 1: Fused GroupNorm, Tanh, simplified HardSwish, and Residual Addition.
// Each block processes one (sample, group) pair.
__global__ void fused_groupnorm_activation_kernel(
    const float* __restrict__ x,       // [N, C, H, W] from convolution.
    float* __restrict__ out,           // same shape as x.
    const float* __restrict__ weight,  // GroupNorm gamma, shape [C]
    const float* __restrict__ bias,    // GroupNorm beta, shape [C]
    int N, int C, int H, int W,
    int groups, float eps)
{
    int channels_per_group = C / groups;
    int hw = H * W;
    int group_size = channels_per_group * hw;

    // Determine sample and group.
    int n = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    
    int start_c = g * channels_per_group;
    int base_idx = n * (C * hw) + start_c * hw;
    const float* group_ptr = x + base_idx;

    // ----------------------------------------------------------------
    // Phase 1: Compute sum and sum-of-squares for the group.
    float local_sum = 0.f, local_sumsq = 0.f;
    if ((group_size & 3) == 0) {
        int vec_size = group_size >> 2;
        const float4* group_ptr4 = reinterpret_cast<const float4*>(group_ptr);
        for (int i = tid; i < vec_size; i += blockDim.x) {
            float4 v = group_ptr4[i];
            local_sum   += v.x + v.y + v.z + v.w;
            local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        for (int i = tid; i < group_size; i += blockDim.x) {
            float t = group_ptr[i];
            local_sum   += t;
            local_sumsq += t*t;
        }
    }

    local_sum = warpReduceSum(local_sum);
    local_sumsq = warpReduceSum(local_sumsq);

    // Use shared memory for inter-warp reduction.
    extern __shared__ float sdata[];  // first half: sums, second half: sumsq.
    int numWarps = (blockDim.x + 31) / 32;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        sdata[warp_id] = local_sum;
        sdata[warp_id + numWarps] = local_sumsq;
    }
    __syncthreads();

    float sum_total = (tid < numWarps) ? sdata[tid] : 0.f;
    float sumsq_total = (tid < numWarps) ? sdata[tid + numWarps] : 0.f;
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < numWarps; i++) {
            sum_total += sdata[i];
            sumsq_total += sdata[i + numWarps];
        }
        float mean = sum_total / group_size;
        float var = sumsq_total / group_size - mean * mean;
        float inv_std = rsqrtf(var + eps);
        // Store mean and inv_std in shared memory.
        sdata[0] = mean;
        sdata[1] = inv_std;
    }
    __syncthreads();
    float mean = sdata[0];
    float inv_std = sdata[1];

    // ----------------------------------------------------------------
    // Phase 2: Per-element normalization (with fused affine transform), activation, and residual.
    if ((hw & 3) == 0) {
        // Process vectorized if spatial dimension is a multiple of 4.
        for (int c_local = 0; c_local < channels_per_group; c_local++) {
            int c = start_c + c_local;
            // Precompute fused affine parameters.
            float gamma = __ldg(&weight[c]);
            float beta = __ldg(&bias[c]);
            float scale = gamma * inv_std;
            float shift = beta - mean * scale;
            const float* channel_in = group_ptr + c_local * hw;
            float* channel_out = out + base_idx + c_local * hw;
            int vec_size = hw >> 2;
            const float4* in_vec = reinterpret_cast<const float4*>(channel_in);
            float4* out_vec = reinterpret_cast<float4*>(channel_out);
            for (int i = tid; i < vec_size; i += blockDim.x) {
                float4 v = in_vec[i];
                float orig0 = v.x, orig1 = v.y, orig2 = v.z, orig3 = v.w;
                // Fused normalization and affine transform.
                float y0 = orig0 * scale + shift;
                float y1 = orig1 * scale + shift;
                float y2 = orig2 * scale + shift;
                float y3 = orig3 * scale + shift;
                // Activation: tanh followed by simplified hardswish.
                float z0 = tanhf(y0);
                float z1 = tanhf(y1);
                float z2 = tanhf(y2);
                float z3 = tanhf(y3);
                float hsw0 = z0 * (z0 + 3.0f) * (1.0f/6.0f);
                float hsw1 = z1 * (z1 + 3.0f) * (1.0f/6.0f);
                float hsw2 = z2 * (z2 + 3.0f) * (1.0f/6.0f);
                float hsw3 = z3 * (z3 + 3.0f) * (1.0f/6.0f);
                float4 res;
                res.x = orig0 + hsw0;
                res.y = orig1 + hsw1;
                res.z = orig2 + hsw2;
                res.w = orig3 + hsw3;
                out_vec[i] = res;
            }
        }
    } else {
        // Scalar fallback if vectorized processing is not possible.
        for (int i = tid; i < group_size; i += blockDim.x) {
            int c_local = i / hw;
            int c = start_c + c_local;
            float orig = group_ptr[i];
            float gamma = __ldg(&weight[c]);
            float beta = __ldg(&bias[c]);
            float scale = gamma * inv_std;
            float shift = beta - mean * scale;
            float y = orig * scale + shift;
            float z = tanhf(y);
            float hsw = z * (z + 3.0f) * (1.0f / 6.0f);
            out[base_idx + i] = orig + hsw;
        }
    }
}

// --------------------------------------------------------------------
// Kernel 2: Fused LogSumExp reduction along the channel dimension.
// For each output pixel (n, h, w), compute:
//    log( sum_c exp(x[n,c,h,w] - m) ) + m, where m = max_c(x[n,c,h,w])
__global__ void logsumexp_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total = N * HW;
    if (idx < total) {
        int n = idx / HW;
        int rem = idx % HW;
        int h = rem / W;
        int w = rem % W;
        int base = n * (C * HW);
        int offset = h * W + w;
        float m = -FLT_MAX;
        #pragma unroll
        for (int c = 0; c < C; c++) {
            int index = base + c * HW + offset;
            float val = __ldg(&x[index]);
            m = fmaxf(m, val);
        }
        float sum_exp = 0.f;
        #pragma unroll
        for (int c = 0; c < C; c++) {
            int index = base + c * HW + offset;
            float val = __ldg(&x[index]);
            sum_exp += __expf(val - m);
        }
        int out_index = n * HW + offset;
        out[out_index] = logf(sum_exp) + m;
    }
}

// --------------------------------------------------------------------
// C++ wrapper for the fused GroupNorm+Activation+Residual kernel.
torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int groups, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto out = torch::empty_like(x);
    int channels_per_group = C / groups;
    int hw = H * W;
    int group_size = channels_per_group * hw;
    int threads = group_size < 256 ? group_size : 256;
    dim3 blocks(N, groups);
    int numWarps = (threads + 31) / 32;
    size_t shared_mem = numWarps * 2 * sizeof(float);
    
    fused_groupnorm_activation_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        groups, eps
    );
    return out;
}

// --------------------------------------------------------------------
// C++ wrapper for the fused LogSumExp kernel.
torch::Tensor logsumexp_forward_cuda(torch::Tensor x) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto out = torch::empty({N, 1, H, W}, x.options());
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    logsumexp_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );
    return out;
}
"""

cpp_source = r'''
torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int groups, float eps);
torch::Tensor logsumexp_forward_cuda(torch::Tensor x);
'''

module = load_inline(name='fused_ops_opt2',
                     cpp_sources=cpp_source,
                     cuda_sources=cuda_source,
                     functions=['fused_forward_cuda', 'logsumexp_forward_cuda'],
                     extra_cflags=['-O3'],
                     extra_cuda_cflags=['-O3', '--use_fast_math'],
                     verbose=False)

class Model(nn.Module):
    """
    Optimized Model performing a convolution, fused GroupNorm,
    Tanh, simplified HardSwish, Residual Addition, and fused LogSumExp reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        # These activations are fused inside the CUDA kernel.
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Step 1: Convolution.
        x_conv = self.conv(x)
        # Step 2: Fused GroupNorm + Activation + Residual.
        x_fused = module.fused_forward_cuda(x_conv, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        # Step 3: Fused LogSumExp reduction over channels.
        x_logsumexp = module.logsumexp_forward_cuda(x_fused)
        return x_logsumexp

def get_inputs():
    batch_size = 128
    in_channels = 3
    height, width = 32, 32
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    groups = 8
    return [in_channels, out_channels, kernel_size, groups]

if __name__ == "__main__":
    batch_size = 128
    in_channels = 3
    out_channels = 16
    height, width = 32, 32
    kernel_size = 3
    groups = 8

    model = Model(in_channels, out_channels, kernel_size, groups).cuda()
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')
    y = model(x)
    print("Output shape:", y.shape)
