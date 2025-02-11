# level 2 index 7 agent name: KernelAgent O3 Mini High speedup: 1.83x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs, "N_BLOCKS": nb, "num_warps": nw})
        for bs in [64, 128, 256, 512, 1024]
        for nb in [1, 2, 4, 8]
        for nw in [4, 8]
    ],
    key=["numel"]
)
@triton.jit
def fused_activation_kernel(
    x_ptr, bias_ptr, output_ptr, numel: tl.constexpr,
    # Full 5D shape of x (assumed contiguous in row‐major order)
    X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr, X4: tl.constexpr,
    # Effective bias shape and strides (for broadcasting); for dimensions that are 1, the stride is forced to 0.
    B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, B4: tl.constexpr,
    BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr, BS3: tl.constexpr, BS4: tl.constexpr,
    # Meta-parameters provided by the autotuner.
    BLOCK_SIZE: tl.constexpr, N_BLOCKS: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute the base offset for this program (each program handles N_BLOCKS blocks of BLOCK_SIZE elements)
    base = pid * BLOCK_SIZE * N_BLOCKS
    for i in range(N_BLOCKS):
        o = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = o < numel

        # Convert the flattened index o into the 5D indices corresponding to x.
        i4 = o % X4
        r = o // X4
        i3 = r % X3
        r = r // X3
        i2 = r % X2
        r = r // X2
        i1 = r % X1
        i0 = r // X1

        # For broadcast dimensions in the bias, if the effective dimension is 1, force index 0.
        b0 = tl.where(B0 == 1, 0, i0)
        b1 = tl.where(B1 == 1, 0, i1)
        b2 = tl.where(B2 == 1, 0, i2)
        b3 = tl.where(B3 == 1, 0, i3)
        b4 = tl.where(B4 == 1, 0, i4)
        bias_index = b0 * BS0 + b1 * BS1 + b2 * BS2 + b3 * BS3 + b4 * BS4

        # Load the convolution output (stored in half precision) and cast to float32 for computation.
        x_val = tl.load(x_ptr + o, mask=mask, other=0.0)
        x_val = tl.cast(x_val, tl.float32)

        # Apply ReLU.
        x_val = tl.maximum(x_val, 0.0)

        # Compute x cube.
        x_cube = x_val * x_val * x_val

        # Simplified GELU approximation:
        # gelu(x) = x * sigmoid(2*(0.7978845608028654*x + 0.0356775*x^3))
        gelu_alpha2 = 1.59576912160573   # 2 * 0.7978845608028654
        gelu_beta2 = 0.071355             # 2 * 0.0356775
        inner = gelu_alpha2 * x_val + gelu_beta2 * x_cube
        gelu_val = x_val * tl.sigmoid(inner)
        x_val = gelu_val

        # Then apply Sigmoid.
        x_val = tl.sigmoid(x_val)

        # Load the bias (in half precision), cast to float32, and add to x.
        bias_val = tl.load(bias_ptr + bias_index, mask=mask, other=0.0)
        bias_val = tl.cast(bias_val, tl.float32)
        x_val = x_val + bias_val

        # Store the result as half precision.
        tl.store(output_ptr + o, tl.cast(x_val, tl.float16), mask=mask)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        # Create a 3D convolution and an extra bias parameter, matching the original module.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Convert convolution weights and bias to half precision.
        self.conv = self.conv.half()
        self.bias = nn.Parameter(self.bias.data.half())

    def forward(self, x):
        # Convert input to half-precision.
        x = x.half()
        # Compute the 3D convolution.
        x = self.conv(x)
        x = x.contiguous()  # ensure x is contiguous

        out = torch.empty_like(x)
        numel = x.numel()
        # x is assumed to be 5D with shape (X0, X1, X2, X3, X4).
        X0, X1, X2, X3, X4 = x.shape

        # Prepare effective bias shape and strides for broadcasting.
        bias_tensor = self.bias
        if bias_tensor.dim() < 5:
            pad = 5 - bias_tensor.dim()
            eff_bias_shape = (1,) * pad + bias_tensor.shape
            eff_bias_stride = (0,) * pad + bias_tensor.stride()
        else:
            eff_bias_shape = bias_tensor.shape
            eff_bias_stride = bias_tensor.stride()
        # For broadcast dimensions, enforce a stride of 0.
        eff_bias_stride = tuple(s if d != 1 else 0 for d, s in zip(eff_bias_shape, eff_bias_stride))

        # Define grid configuration using the autotuner's meta-parameters.
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"] * meta["N_BLOCKS"]),)
        fused_activation_kernel[grid](
            x,                   # x_ptr: pointer to the convolution output (half precision)
            bias_tensor,         # bias_ptr: pointer to the bias (half precision)
            out,                 # output_ptr: pointer to the output (half precision)
            numel,
            X0, X1, X2, X3, X4,  # full shape of x
            eff_bias_shape[0], eff_bias_shape[1], eff_bias_shape[2], eff_bias_shape[3], eff_bias_shape[4],
            eff_bias_stride[0], eff_bias_stride[1], eff_bias_stride[2], eff_bias_stride[3], eff_bias_stride[4]
        )

        # Cast the final result back to float32.
        return out.float()
