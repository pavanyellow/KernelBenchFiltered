# level 2 index 59 agent name: KernelAgent o1 speedup: 1.55x

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _matmul_sigmoid_scale_kernel(
    x_ptr,         # [M, K]
    w_ptr,         # [N, K]
    b_ptr,         # [N]
    out_ptr,       # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_b,
    stride_om, stride_on,
    scaling_factor,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    """
    Computes out = (x @ w.T) * sigmoid(x @ w.T) * scaling_factor + bias_applied
    - x: [M, K]
    - w: [N, K]  (since weight is shape [out_features, in_features])
    - bias: [N]
    - output: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Range of rows/columns this program handles
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Partial accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        # Load x sub-block: shape [BLOCK_M, BLOCK_K]
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        x_tile = tl.load(
            x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk),
            mask=x_mask,
            other=0.0
        )

        # Load w sub-block: shape [BLOCK_N, BLOCK_K]
        w_mask = (rn[:, None] < N) & (rk[None, :] < K)
        w_tile = tl.load(
            w_ptr + (rn[:, None] * stride_wn + rk[None, :] * stride_wk),
            mask=w_mask,
            other=0.0
        )
        # Transpose w_tile so it becomes [BLOCK_K, BLOCK_N]
        w_tile = tl.trans(w_tile)

        # Dot: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_tile, w_tile)

    # Add bias, apply sigmoid and scaling
    bias_mask = rn < N
    bias_vals = tl.load(b_ptr + rn * stride_b, mask=bias_mask, other=0.0)
    acc += bias_vals[None, :]

    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-acc))
    # multiply by own sigmoid
    acc = acc * sig
    # scale
    acc *= scaling_factor

    # Store result
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        out_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on),
        acc,
        mask=out_mask
    )


class Model(nn.Module):
    """
    This Triton-accelerated model has the same interface, same default parameter
    initialization, and produces the same outputs (within float tolerance)
    as the original code:

        x = self.matmul(x)
        x = x * torch.sigmoid(x)
        x = x * self.scaling_factor
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # Use an nn.Linear just to replicate Torch's default init
        ref_linear = nn.Linear(in_features, out_features)
        self.weight = ref_linear.weight  # [out_features, in_features]
        self.bias = ref_linear.bias      # [out_features]
        self.scaling_factor = scaling_factor

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor):
        """
        x shape: [batch_size, in_features]
        output shape: [batch_size, out_features]
        For our specific case: (128,1024) -> (128,512)
        """
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Launch grid
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 64
        grid = (
            (M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N
        )

        _matmul_sigmoid_scale_kernel[grid](
            x,
            self.weight,
            self.bias,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0),
            out.stride(0), out.stride(1),
            self.scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )
        return out
