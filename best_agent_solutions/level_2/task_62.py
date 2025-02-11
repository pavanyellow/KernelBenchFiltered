# level 2 index 62 agent name: KernelAgent o1 speedup: 1.79x

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _groupnorm_leakyrelu_double_kernel(
    x_ptr,          # *float32
    out_ptr,        # *float32
    gamma_ptr,      # *float32
    beta_ptr,       # *float32
    N,              # int
    C,              # int
    G,              # int
    group_size,     # int
    eps,            # float
    negative_slope, # float
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel that fuses GroupNorm + LeakyReLU + doubling in one pass.
    Each block processes exactly one (batch_element b, group g) pair.
    We assume group_size <= BLOCK_SIZE (so we launch with block_size=group_size).

    Inputs:
      x:     [N, C] float32
      gamma: [C]    float32
      beta:  [C]    float32
      N: number of rows (batch elements)
      C: total number of channels
      G: number of groups
      group_size = C // G
      eps for numerical stability
      negative_slope for LeakyReLU
    """
    # The block index in [0 .. N*G-1]
    b_g = tl.program_id(0)
    # Which batch element and which group?
    b_id = b_g // G
    g_id = b_g % G

    # Each thread in the block processes exactly 1 channel index within [0..group_size-1].
    offsets = tl.arange(0, BLOCK_SIZE)
    # Base offset for memory in x/out/gamma/beta
    c_start = g_id * group_size
    # Compute absolute channel indices
    c_idx = c_start + offsets

    # Base pointer for x/out for this row b_id
    row_start = b_id * C

    # 1) Load x into 'val'
    #    Since group_size <= BLOCK_SIZE for these calls, this is a direct one-to-one mapping.
    x_offset = row_start + c_idx
    val = tl.load(x_ptr + x_offset)

    # 2) First pass: compute sum and sum of squares across the group
    sum_val = tl.sum(val, axis=0)
    sum_sq_val = tl.sum(val * val, axis=0)

    # 3) Compute mean, var, rstd
    #    Each block handles exactly one group, so we can do this in scalar form for that group.
    denom = tl.cast(group_size, tl.float32)
    mean = sum_val / denom
    var = (sum_sq_val / denom) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # 4) Load gamma, beta
    g_val = tl.load(gamma_ptr + c_idx)
    b_val = tl.load(beta_ptr + c_idx)

    # 5) Normalize and apply gamma/beta
    val = (val - mean) * rstd
    val = val * g_val + b_val

    # 6) Fused LeakyReLU + doubling:
    #    If val < 0 =>  val *= (2 * negative_slope)
    #    else       =>  val *= 2
    neg_mask = val < 0.0
    val_pos = val * 2.0
    val_neg = val * (2.0 * negative_slope)
    val = tl.where(neg_mask, val_neg, val_pos)

    # 7) Store results
    tl.store(out_ptr + x_offset, val)


def _groupnorm_leakyrelu_double_triton(x, gamma, beta, eps, negative_slope, groups):
    """
    Python entry point that launches the above Triton kernel to fuse GroupNorm + LeakyReLU + doubling.
    x:     (N, C) float32
    gamma: (C,)   float32
    beta:  (C,)   float32
    """
    assert x.is_cuda, "Input x must be a CUDA tensor."
    N, C = x.shape
    assert C % groups == 0, "C must be divisible by the number of groups"
    group_size = C // groups

    # Allocate output
    out = torch.empty_like(x)

    # Each block handles exactly one (batch_index, group_index) pair.
    grid = (N * groups,)

    # We require group_size <= block_size for the kernel. In the specified usage, group_size=32.
    # If you anticipate bigger group_size, expand the kernel or add loops.
    BLOCK_SIZE = group_size

    _groupnorm_leakyrelu_double_kernel[grid](
        x, out, gamma, beta,
        N, C, groups, group_size,
        eps, negative_slope,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        # Keep the same layers so that initialization (including random init) remains the same.
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        # 1) Run standard PyTorch matmul for fully-connected
        x = self.fc(x)
        # 2) Then do the fused groupnorm+leakyrelu+double in Triton
        x = _groupnorm_leakyrelu_double_triton(
            x,
            self.gn.weight,
            self.gn.bias,
            self.gn.eps,
            self.leaky_relu.negative_slope,
            self.gn.num_groups
        )
        return x
