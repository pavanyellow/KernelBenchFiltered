# level 1 index 35 agent name: KernelAgent O3 Mini High speedup: 2.40x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Kernel 1: Partial reduction over each tile of 1024 contiguous elements.
@triton.jit
def partial_reduction_kernel(x_ptr, partial_ptr,
                             N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                             num_groups: tl.constexpr, channels_per_group: tl.constexpr,
                             TILE_SIZE: tl.constexpr):
    # Each group’s data is contiguous and has group_size = channels_per_group * H * W.
    group_size = channels_per_group * H * W  # e.g. 8 * 256 * 256 = 524288.
    # Each group is split into NUM_TILES = group_size / TILE_SIZE.
    NUM_TILES = group_size // TILE_SIZE  # should be 524288/1024 = 512.
    # Each kernel instance processes one tile.
    pid = tl.program_id(0)
    # Compute which sample and which group/tile within that sample.
    tiles_per_sample = num_groups * NUM_TILES
    sample_idx = pid // tiles_per_sample
    tmp = pid % tiles_per_sample
    group_idx = tmp // NUM_TILES
    tile_idx = tmp % NUM_TILES
    # Compute the base offset for this group in x (x is in NCHW layout).
    group_base = sample_idx * (C * H * W) + (group_idx * channels_per_group) * (H * W)
    tile_offset = tile_idx * TILE_SIZE
    offset = group_base + tile_offset
    # Load TILE_SIZE contiguous elements.
    offsets = tl.arange(0, TILE_SIZE)
    vals = tl.load(x_ptr + offset + offsets)
    # Compute partial sum and partial sum-of-squares.
    psum   = tl.sum(vals, axis=0)
    psumsq = tl.sum(vals * vals, axis=0)
    # Store two float values per tile.
    tl.store(partial_ptr + pid * 2,     psum)
    tl.store(partial_ptr + pid * 2 + 1, psumsq)

    
# Kernel 2: Final reduction per group over the tiles.
@triton.jit
def final_reduction_kernel(partial_ptr, stats_ptr,
                           NUM_TILES: tl.constexpr, group_size: tl.constexpr):
    # One kernel instance per group (across a sample).
    group_id = tl.program_id(0)
    offsets = tl.arange(0, NUM_TILES)
    # Partial results for this group are stored consecutively.
    base = group_id * (NUM_TILES * 2)
    partial_sums  = tl.load(partial_ptr + base + offsets * 2)
    partial_sumsq = tl.load(partial_ptr + base + offsets * 2 + 1)
    total_sum   = tl.sum(partial_sums, axis=0)
    total_sumsq = tl.sum(partial_sumsq, axis=0)
    mean = total_sum / group_size
    var  = total_sumsq / group_size - mean * mean
    tl.store(stats_ptr + group_id * 2,     mean)
    tl.store(stats_ptr + group_id * 2 + 1, var)

    
# Kernel 3: Elementwise normalization and affine transformation.
# We add autotuning to vary two key parameters:
#   BLOCK_SIZE: number of contiguous elements processed per inner loop iteration.
#   NUM_BLOCKS: number of such blocks processed per kernel instance.
# We provide about five power-of-2 choices for BLOCK_SIZE between 64 and 2048,
# autotuning NUM_BLOCKS over [1, 2, 4, 8], and try num_warps of 4 or 8.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs, "NUM_BLOCKS": nb}, num_warps=w)
        for bs in [64, 256, 512, 1024, 2048]
        for nb in [1, 2, 4, 8]
        for w in [4, 8]
    ],
    key=["total_elems"],
)
@triton.jit
def norm_kernel(x_ptr, stats_ptr, weight_ptr, bias_ptr, y_ptr,
                N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                num_groups: tl.constexpr, channels_per_group: tl.constexpr, eps: tl.constexpr,
                total_elems: tl.constexpr,
                BLOCK_SIZE: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    # Each kernel instance processes NUM_BLOCKS contiguous blocks of BLOCK_SIZE elements.
    base_idx = tl.program_id(0) * (BLOCK_SIZE * NUM_BLOCKS)
    # Loop over the blocks handled by this program instance.
    for b in range(NUM_BLOCKS):
        offset = base_idx + b * BLOCK_SIZE
        idx = offset + tl.arange(0, BLOCK_SIZE)
        # Since we assume total number of elements is an exact multiple of (BLOCK_SIZE*NUM_BLOCKS),
        # we do not need to use masks.
        x_vals = tl.load(x_ptr + idx)
        # Compute per-element indices in the flattened (N, C, H, W) tensor.
        CHW = C * H * W
        HW  = H * W
        n   = idx // CHW           # Sample index.
        rem = idx % CHW
        c   = rem // HW            # Channel index.
        # Determine the group index given that channels within a group are contiguous.
        g = c // channels_per_group
        # For sample n and group g, the corresponding statistics (mean, var) are stored at:
        group_global = n * num_groups + g
        mean = tl.load(stats_ptr + group_global * 2)
        var  = tl.load(stats_ptr + group_global * 2 + 1)
        inv_std = tl.rsqrt(var + eps)
        norm_val = (x_vals - mean) * inv_std
        # Load per-channel affine parameters.
        w = tl.load(weight_ptr + c)
        b_val = tl.load(bias_ptr + c)
        out_val = norm_val * w + b_val
        tl.store(y_ptr + idx, out_val)

        
# The optimized GroupNorm model with the same interface as the original Model.
class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.channels_per_group = num_features // num_groups
        self.eps = 1e-5
        # Learned per-channel affine parameters (initialized as in GroupNorm).
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume input x has shape (16, 64, 256, 256) and is contiguous.
        N, C, H, W = x.shape  # N=16, C=64, H=256, W=256.
        channels_per_group = self.channels_per_group
        # Each group has group_size = channels_per_group * H * W.
        group_size = channels_per_group * H * W  # e.g., 8 * 256 * 256.
        # Reduction is tiled in chunks of 1024 elements.
        RED_TILE_SIZE = 1024
        NUM_TILES = group_size // RED_TILE_SIZE  # Should be exactly 512.
        # Allocate temporary buffer for partial reductions.
        total_tiles = N * self.num_groups * NUM_TILES
        partial_buffer = torch.empty(total_tiles * 2, dtype=torch.float32, device=x.device)
        # Allocate a stats buffer to hold mean and variance for each group.
        stats_buffer = torch.empty(N * self.num_groups * 2, dtype=torch.float32, device=x.device)

        # Kernel 1: Launch partial reduction kernel (one program per tile).
        grid1 = (total_tiles,)
        partial_reduction_kernel[grid1](x, partial_buffer,
                                        N, C, H, W,
                                        self.num_groups, channels_per_group,
                                        RED_TILE_SIZE)

        # Kernel 2: Final reduction kernel (one program per group; grid size = N * num_groups).
        grid2 = (N * self.num_groups,)
        final_reduction_kernel[grid2](partial_buffer, stats_buffer, NUM_TILES, group_size)

        # Kernel 3: Elementwise normalization and affine transformation.
        y = torch.empty_like(x)
        total_elems = N * C * H * W
        # Use a grid lambda so that the autotuner can provide the tuned BLOCK_SIZE and NUM_BLOCKS.
        grid3 = lambda META: (triton.cdiv(total_elems, META["BLOCK_SIZE"] * META["NUM_BLOCKS"]),)
        # The norm_kernel is autotuned over BLOCK_SIZE (per-block work) and NUM_BLOCKS (blocks per kernel)
        # together with num_warps.
        norm_kernel[grid3](x, stats_buffer, self.weight, self.bias, y,
                           N, C, H, W,
                           self.num_groups, channels_per_group, self.eps,
                           total_elems)
        return y
