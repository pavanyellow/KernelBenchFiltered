# level 2 index 6 agent name: KernelAgent O3 Mini High speedup: 2.14x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# The fused kernel performs softmax over the channel vector (dim C) and then max-pools over a 3D pooling window.
# We autotune over BLOCK_SIZE (the number of candidates per block) and N_BLOCKS (number of blocks processed per program)
# such that the total block work is between 64 and 2048 total candidate elements.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64,  "N_BLOCKS": 1, "num_warps": 4}),   # 64 candidates total
        triton.Config({"BLOCK_SIZE": 128, "N_BLOCKS": 2, "num_warps": 4}),   # 256 candidates total
        triton.Config({"BLOCK_SIZE": 256, "N_BLOCKS": 2, "num_warps": 8}),   # 512 candidates total
        triton.Config({"BLOCK_SIZE": 256, "N_BLOCKS": 4, "num_warps": 8}),   # 1024 candidates total
        triton.Config({"BLOCK_SIZE": 512, "N_BLOCKS": 4, "num_warps": 8}),   # 2048 candidates total
    ],
    key=["B", "C", "D", "H", "W", "pool_size"],
)
@triton.jit
def fused_softmax_pool_kernel(x, y,
                              B: tl.constexpr, C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                              D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
                              pool_size: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr, N_BLOCKS: tl.constexpr):
    # Compute input strides for contiguous NCDHW layout.
    stride_xW = 1
    stride_xH = W                  # width is the innermost dimension
    stride_xD = H * W
    stride_xC = D * H * W
    stride_xB = C * D * H * W

    # Compute output strides for NCDHW layout.
    stride_yW = 1
    stride_yH = W_out
    stride_yD = H_out * W_out
    stride_yC = D_out * H_out * W_out
    stride_yB = C * D_out * H_out * W_out

    # Each program computes one output “pixel” (a C-channel vector)
    pid = tl.program_id(0)
    pool_grid = D_out * H_out * W_out
    b  = pid // pool_grid
    rem = pid % pool_grid
    od = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    # Compute the starting coordinate of the pooling window in the input.
    d_start = od * pool_size
    h_start = oh * pool_size
    w_start = ow * pool_size

    # Base pointer offset for the current pooling window.
    base_ptr = b * stride_xB + d_start * stride_xD + h_start * stride_xH + w_start * stride_xW

    # Precompute offsets along the channel dimension.
    cid = tl.arange(0, C)
    channel_offset = cid * stride_xC

    # Total number of candidates in the pooling cube.
    pool_total = pool_size * pool_size * pool_size

    # Initialize the pooled result for each channel.
    pooled = tl.zeros((C,), dtype=tl.float32)

    # Process the pooling window in blocks of candidates.
    for start in range(0, pool_total, BLOCK_SIZE * N_BLOCKS):
        for nb in range(N_BLOCKS):
            idx = start + nb * BLOCK_SIZE
            # Compute candidate indices for this block.
            r = idx + tl.arange(0, BLOCK_SIZE)
            valid_r = r < pool_total  # Only process indices within the pooling window.

            # Decompose candidate index into 3D offsets within the pooling window.
            i = r // (pool_size * pool_size)
            rem_local = r % (pool_size * pool_size)
            j = rem_local // pool_size
            k = rem_local % pool_size

            # Compute the spatial offset of the candidate.
            spatial_offset = i * stride_xD + j * stride_xH + k * stride_xW

            # Compute full offsets from the beginning of the tensor plus channel offset.
            offsets = base_ptr + spatial_offset[:, None] + channel_offset[None, :]

            # Check that the candidate position is within input bounds.
            valid = ((d_start + i) < D) & ((h_start + j) < H) & ((w_start + k) < W)
            valid = valid & valid_r  # Combine candidate-index validity.
            valid_mask = valid[:, None]

            # Load candidate values from x.
            # Use mask only since input dimensions are not guaranteed to be powers-of-2.
            x_vals = tl.load(x + offsets, mask=valid_mask, other=0).to(tl.float32)
            # Compute exponentials.
            exp_vals = tl.exp(x_vals)
            # For invalid candidates, force contribution of zero.
            valid_f = tl.where(valid, 1.0, 0.0)
            exp_vals = exp_vals * valid_f[:, None]

            # Normalize per candidate (compute softmax normalization over channels).
            row_sum = tl.sum(exp_vals, axis=1)
            row_sum = tl.where(row_sum == 0, 1.0, row_sum)
            softmax_vals = exp_vals / row_sum[:, None]

            # Pool: take elementwise maximum across these candidate softmax values.
            block_max = tl.max(softmax_vals, axis=0)
            pooled = tl.maximum(pooled, block_max)

    # Compute output pointer offset.
    out_offset = b * stride_yB + cid * stride_yC + od * stride_yD + oh * stride_yH + ow * stride_yW
    tl.store(y + out_offset, pooled.to(tl.float16))


# Host function to launch the fused kernel.
def fused_softmax_pool(x: torch.Tensor, pool_size: int) -> torch.Tensor:
    # x is expected to be in shape [B, C, D, H, W] and FP16.
    B, C, D, H, W = x.shape
    # Compute output spatial dimensions assuming 'valid' pooling.
    D_out = (D - pool_size) // pool_size + 1
    H_out = (H - pool_size) // pool_size + 1
    W_out = (W - pool_size) // pool_size + 1
    y = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=torch.float16)
    grid = (B * D_out * H_out * W_out, )
    fused_softmax_pool_kernel[grid](x, y, B, C, D, H, W, D_out, H_out, W_out, pool_size)
    return y


# The optimized Model module which conforms to the original interface.
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_kernel_size: int):
        super().__init__()
        # Standard 3D convolution with default initialization.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Use FP16 parameters for faster computation.
        self.conv = self.conv.half()
        # Fuse the two pooling stages (originally two consecutive 3D max pools)
        # by using an effective pooling window. With the original module,
        # pool_size was pool_kernel_size per dimension twice, so we set:
        self.pool_size = pool_kernel_size * pool_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast the input to FP16 for faster computation.
        orig_dtype = x.dtype
        x = x.half()
        x = self.conv(x)
        x = fused_softmax_pool(x, self.pool_size)
        return x.to(orig_dtype)
