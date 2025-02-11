# level 2 index 35 agent name: KernelAgent O3 Mini High speedup: 1.55x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source:
# This kernel fuses, for each pooling window:
#   - subtracting a constant,
#   - applying a fused hardswish activation,
#   - reducing via max pool over the window, and then
#   - applying the mish activation.
#
# There are two kernel variants:
# 1. A generic kernel that works for any pool_size.
# 2. A specialized kernel for pool_size == 2 when H and W are divisible by 2,
#    which uses vectorized loads (float2) for improved memory throughput.
#
# Each output element (at indices (n, c, h_out, w_out)) depends only on a small
# pooling window of the convolution output; no cross-thread communication is required.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------------------
// Fused hardswish:
// First subtract sub_val then compute hardswish:
//   v = x - sub_val
//   t = clamp(v + 3, 0, 6)
//   activated = v * t/6
__device__ inline float fused_hardswish_fn(float x, float sub_val) {
    x = x - sub_val;
    float t = x + 3.0f;
    t = fminf(fmaxf(t, 0.0f), 6.0f);
    return x * t * 0.166666667f;  // Multiply by 1/6
}

// ------------------------------
// Mish activation:
//   mish(x) = x * tanh(log(1+exp(x)))
__device__ inline float mish_fn(float x) {
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

// ------------------------------
// Generic kernel for arbitrary pool_size.
// For each output element (n, c, h_out, w_out):
//   - Determine pooling window start: (h_start = h_out * pool_size, w_start = w_out * pool_size)
//   - Loop over all pool_size×pool_size positions, applying fused hardswish to each input,
//     then reduce via max.
//   - Finally, apply mish activation on the pooled max.
__global__ void fused_pool_act_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int N, int C, int H, int W,
                                        int pool_size, int out_H, int out_W,
                                        float sub_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    if (idx >= total)
        return;

    // Decode linear index to (n, c, h_out, w_out)
    int w_out = idx % out_W;
    int tmp = idx / out_W;
    int h_out = tmp % out_H;
    int tmp2 = tmp / out_H;
    int c = tmp2 % C;
    int n = tmp2 / C;

    int h_start = h_out * pool_size;
    int w_start = w_out * pool_size;
    float max_val = -1e30f;

    // Loop over the pooling window.
    for (int i = 0; i < pool_size; i++) {
        for (int j = 0; j < pool_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in < H && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                float val = __ldg(&input[in_idx]);
                float activated = fused_hardswish_fn(val, sub_val);
                if (activated > max_val)
                    max_val = activated;
            }
        }
    }
    // Apply mish activation.
    output[idx] = mish_fn(max_val);
}

// ------------------------------
// Specialized kernel for pool_size == 2.
// When H and W are divisible by 2, we can use vectorized loads (float2)
// to read 2 contiguous floats in one go. The dependencies remain identical.
__global__ void fused_pool_act_kernel_2(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int N, int C, int H, int W,
                                          int out_H, int out_W,
                                          float sub_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    if (idx >= total)
        return;

    // Decode (n, c, h_out, w_out)
    int w_out = idx % out_W;
    int tmp = idx / out_W;
    int h_out = tmp % out_H;
    int tmp2 = tmp / out_H;
    int c = tmp2 % C;
    int n = tmp2 / C;

    int h_start = h_out * 2;
    int w_start = w_out * 2;
    int base = ((n * C + c) * H + h_start) * W + w_start;

    // Load two rows of the 2x2 pooling window as float2 vectors.
    const float2* row_ptr0 = reinterpret_cast<const float2*>(&input[base]);
    const float2* row_ptr1 = reinterpret_cast<const float2*>(&input[base + W]);
    float2 row0 = __ldg(row_ptr0);
    float2 row1 = __ldg(row_ptr1);

    float a = fused_hardswish_fn(row0.x, sub_val);
    float b = fused_hardswish_fn(row0.y, sub_val);
    float c_val = fused_hardswish_fn(row1.x, sub_val);
    float d = fused_hardswish_fn(row1.y, sub_val);

    float max_val = a;
    max_val = (b > max_val ? b : max_val);
    max_val = (c_val > max_val ? c_val : max_val);
    max_val = (d > max_val ? d : max_val);

    output[idx] = mish_fn(max_val);
}

// ------------------------------
// Dispatch function: calculates the dimensions of the output tensor and
// launches the appropriate kernel variant based on pool_size and input shape.
torch::Tensor fused_pool_act_cuda_v2(torch::Tensor input, int pool_size, float sub_val) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    input = input.contiguous();
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    int out_H, out_W;
    if (H % pool_size == 0 && W % pool_size == 0) {
        out_H = H / pool_size;
        out_W = W / pool_size;
    } else {
        out_H = (H - pool_size) / pool_size + 1;
        out_W = (W - pool_size) / pool_size + 1;
    }

    auto output = torch::empty({N, C, out_H, out_W}, input.options());
    int total = N * C * out_H * out_W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (pool_size == 2 && (H % 2 == 0) && (W % 2 == 0)) {
        fused_pool_act_kernel_2<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            N, C, H, W,
            out_H, out_W, sub_val
        );
    } else {
        fused_pool_act_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            N, C, H, W,
            pool_size, out_H, out_W, sub_val
        );
    }
    return output;
}
"""

# Minimal C++ declaration.
cpp_source = r"""
#include <torch/extension.h>
torch::Tensor fused_pool_act_cuda_v2(torch::Tensor input, int pool_size, float sub_val);
"""

# Compile the inline CUDA/C++ extension.
fused_module = load_inline(
    name="fused_pool_act_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_pool_act_cuda_v2"],
    verbose=False
)

# -----------------------------------------------------------------
# The Optimized Model with the same interface as the original.
#
# It performs a convolution, then fuses subtraction, hardswish activation,
# pooling, and mish activation. If the convolution layer has a bias, the
# subtraction constant is fused into the bias.
# -----------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # If bias exists, fuse the subtraction into the bias.
        if self.conv.bias is not None:
            with torch.no_grad():
                self.conv.bias.sub_(subtract_value)
            self.subtract_value = None  # Already fused into the bias.
        else:
            self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Execute the convolution (cuDNN optimized).
        x = self.conv(x)
        # Use axis-specific subtraction only if bias was not fused.
        sub = self.subtract_value if self.subtract_value is not None else 0.0
        # Launch the fused CUDA operator: subtraction, hardswish, pooling, then mish.
        x = fused_module.fused_pool_act_cuda_v2(x, self.pool_kernel_size, sub)
        return x

# -----------------------------------------------------------------
# Helper functions for testing (not part of the module API)
# -----------------------------------------------------------------
def get_init_inputs():
    # Returns (in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size)
    return (3, 16, 3, 0.5, 2)

def get_inputs():
    # Returns a typical input tensor: [batch_size, channels, height, width]
    return [torch.randn(128, 3, 32, 32, dtype=torch.float32)]

# -----------------------------------------------------------------
# Test code (runs when executed as a script)
# -----------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(3, 16, 3, 0.5, 2).to(device)
    x = torch.randn(128, 3, 32, 32, device=device, dtype=torch.float32)
    # Warm-up iterations.
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    out = model(x)
    torch.cuda.synchronize()
    print("Output shape:", out.shape)
