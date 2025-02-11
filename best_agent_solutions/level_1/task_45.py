# level 1 index 45 agent name: KernelAgent o1 speedup: 1.91x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# For the case kernel_size=3, stride=3, padding=0, we can omit boundary checks and unroll the 3x3 summation
# because each output cell maps to a disjoint 3x3 region in the input.

_avgpool2d_cuda_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Specialized average pooling for kernel_size=3, stride=3, padding=0.
// We rely on the fact that for outH = (H - 3)/3 + 1 and outW = (W - 3)/3 + 1,
// each 3x3 tile is fully in-bounds (since H=256, W=256, outH=85, outW=85).
// No boundary checks are required.
////////////////////////////////////////////////////////////////////////////////
__global__ void avgpool2d_3x3s3p0_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C,
    const int H, const int W,
    const int outH, const int outW)
{
    // Each thread produces one output element: (n, c, oh, ow).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    // Decompose index
    int n  = idx / (C * outH * outW);
    int r1 = idx % (C * outH * outW);
    int c0 = r1 / (outH * outW);
    int r2 = r1 % (outH * outW);
    int oh = r2 / outW;
    int ow = r2 % outW;

    // Compute the top-left corner of the 3x3 region in the input
    // (no padding, stride=3)
    int h_in = oh * 3;
    int w_in = ow * 3;

    // Base pointer offset for (n, c, h_in, w_in)
    int base_in = ((n * C + c0) * H + h_in) * W + w_in;

    // Sum up the 3x3 patch
    float sum_val = 0.0f;
    sum_val += input[base_in + 0*W + 0];
    sum_val += input[base_in + 0*W + 1];
    sum_val += input[base_in + 0*W + 2];
    sum_val += input[base_in + 1*W + 0];
    sum_val += input[base_in + 1*W + 1];
    sum_val += input[base_in + 1*W + 2];
    sum_val += input[base_in + 2*W + 0];
    sum_val += input[base_in + 2*W + 1];
    sum_val += input[base_in + 2*W + 2];

    // Divide by 9.0
    output[idx] = sum_val / 9.0f;
}


////////////////////////////////////////////////////////////////////////////////
// Generic fallback kernel if needed. This is the same naive code from earlier,
// but in this example, we know we will only call it with kernel=3, stride=3, padding=0.
// We'll keep it here for completeness.
////////////////////////////////////////////////////////////////////////////////
__global__ void avgpool2d_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C,
    const int H, const int W,
    const int outH, const int outW,
    const int kernel_size,
    const int stride,
    const int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int n  = idx / (C * outH * outW);
    int r1 = idx % (C * outH * outW);
    int c0 = r1 / (outH * outW);
    int r2 = r1 % (outH * outW);
    int oh = r2 / outW;
    int ow = r2 % outW;

    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    float sum_val = 0.0f;
    for(int kh = 0; kh < kernel_size; kh++){
        int h_in = h_start + kh;
        if(h_in >= 0 && h_in < H) {
            for(int kw = 0; kw < kernel_size; kw++){
                int w_in = w_start + kw;
                if(w_in >= 0 && w_in < W){
                    sum_val += input[((n*C + c0)*H + h_in)*W + w_in];
                }
            }
        }
    }
    float pool_size = static_cast<float>(kernel_size * kernel_size);
    output[idx] = sum_val / pool_size;
}

////////////////////////////////////////////////////////////////////////////////
// The forward function dispatches either the specialized kernel (if kernel=3,
// stride=3, padding=0) or the generic fallback otherwise.
////////////////////////////////////////////////////////////////////////////////
torch::Tensor avgpool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    // Input shape is (N, C, H, W).
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // If stride <= 0, default to kernel_size
    if (stride <= 0) {
        stride = kernel_size;
    }

    // Compute output sizes
    int outH = (H + 2*padding - kernel_size) / stride + 1;
    int outW = (W + 2*padding - kernel_size) / stride + 1;

    auto options = input.options().requires_grad(false);
    torch::Tensor output = torch::empty({N, C, outH, outW}, options);

    int total = N * C * outH * outW;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    // Check if (kernel=3, stride=3, padding=0). If so, use specialized kernel.
    if (kernel_size == 3 && stride == 3 && padding == 0) {
        avgpool2d_3x3s3p0_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            outH, outW
        );
    } else {
        // Fallback to the generic kernel
        avgpool2d_generic_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            outH, outW,
            kernel_size,
            stride,
            padding
        );
    }

    return output;
}
'''


# We declare our C++ "header" to expose avgpool2d_forward:
_avgpool2d_cpp_src = r"""
torch::Tensor avgpool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding);
"""

# Compile the inline CUDA code (specialized + fallback) for our custom average pooling
_avgpool2d_module = load_inline(
    name="custom_avgpool2d",
    cpp_sources=_avgpool2d_cpp_src,
    cuda_sources=_avgpool2d_cuda_src,
    functions=["avgpool2d_forward"],
    verbose=False,
)


class Model(nn.Module):
    """
    This Model replaces nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
    with a custom CUDA-based average pooling implementation. It preserves
    the same interface and output shape/behavior within float tolerance.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        # Store parameters exactly like the original interface
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else 0  # 0 => interpret as "None" inside kernel
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dispatch to our custom CUDA avgpool2d function
        return _avgpool2d_module.avgpool2d_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )
