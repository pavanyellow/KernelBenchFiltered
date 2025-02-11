# level 1 index 40 agent name: KernelAgent o1 speedup: 26.40x

import torch
import torch.nn as nn
import triton
import triton.language as tl

#
# We use a hierarchical reduction in two stages to avoid a very large unrolled loop.
# Stage 1: partial_sums_1_kernel sums chunks of BLOCK_SIZE elements, storing partial sums in ps1.
# Stage 2: partial_sums_2_kernel merges groups of GROUP_SIZE chunk-partials into ps2.
# Stage 3: partial_sums_3_kernel merges those partial sums (which are now much fewer) into a final array ps3.
# Finally, layernorm_kernel uses ps3 to normalize x in chunks, storing the result.
#

@triton.jit
def partial_sums_1_kernel(
    x_ptr,         # float32 ptr to input x, shape = (B, LN), contiguous
    ps1_ptr,       # float32 ptr to partial sums array, shape = (B, LN//BLOCK_SIZE, 2)
    BLOCK_SIZE: tl.constexpr,
    LN: tl.constexpr,   # LN = 64*256*256 = 4194304
    B: tl.constexpr     # B = 16
):
    """
    Each block sums a chunk of size BLOCK_SIZE in x, storing [sum, sum_sq].
    Grid shape is (B, LN//BLOCK_SIZE).
    """
    b_id = tl.program_id(0)         # batch index
    chunk_id = tl.program_id(1)     # which chunk of LN we handle
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offset = b_id * LN + offsets
    # Load chunk of x and compute sums
    x_data = tl.load(x_ptr + x_offset)  # LN & BLOCK_SIZE are powers of 2 -> no mask needed
    sum_ = tl.sum(x_data, axis=0)
    sum_sq_ = tl.sum(x_data * x_data, axis=0)
    # Store sums to ps1_ptr
    # We lay out partial sums as [ (b_id*#chunks + chunk_id), 2 ]
    chunks_per_batch = LN // BLOCK_SIZE
    base_idx = (b_id * chunks_per_batch + chunk_id) * 2
    tl.store(ps1_ptr + base_idx + 0, sum_)
    tl.store(ps1_ptr + base_idx + 1, sum_sq_)


@triton.jit
def partial_sums_2_kernel(
    ps1_ptr,
    ps2_ptr,
    BLOCK_SIZE: tl.constexpr,  
    LN: tl.constexpr,
    B: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """
    Merges groups of GROUP_SIZE partial sums from ps1.
    Grid shape is (B, (#chunks // GROUP_SIZE)).

    For LN=2^22 and BLOCK_SIZE=1024, #chunks = 4096. With GROUP_SIZE=256, we have 4096//256=16,
    so each block merges 256 ps1-sums, leaving 16 partial sums per B for partial_sums_3.
    """
    b_id = tl.program_id(0)
    grp_id = tl.program_id(1)
    chunk_count = LN // BLOCK_SIZE
    base_idx = (b_id * chunk_count + grp_id * GROUP_SIZE) * 2

    sum_ = 0.0
    sum_sq_ = 0.0
    # accumulate partial sums in this group
    for i in range(GROUP_SIZE):
        s = tl.load(ps1_ptr + base_idx + i * 2)
        sq = tl.load(ps1_ptr + base_idx + i * 2 + 1)
        sum_ += s
        sum_sq_ += sq

    # Store result in ps2
    # shape of ps2 is (B * (chunk_count // GROUP_SIZE), 2) flattened
    out_idx = (b_id * (chunk_count // GROUP_SIZE) + grp_id) * 2
    tl.store(ps2_ptr + out_idx, sum_)
    tl.store(ps2_ptr + out_idx + 1, sum_sq_)


@triton.jit
def partial_sums_3_kernel(
    ps2_ptr,
    ps3_ptr,
    BLOCK_SIZE: tl.constexpr,
    LN: tl.constexpr,
    B: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """
    Now merges the results from partial_sums_2 into a final array ps3 (B,2).
    Grid shape is (B).
    """
    b_id = tl.program_id(0)
    chunk_count = LN // BLOCK_SIZE
    partial_count = chunk_count // GROUP_SIZE  # e.g. 4096//256=16

    sum_ = 0.0
    sum_sq_ = 0.0
    base_idx = b_id * partial_count * 2
    for i in range(partial_count):
        s = tl.load(ps2_ptr + base_idx + i * 2)
        sq = tl.load(ps2_ptr + base_idx + i * 2 + 1)
        sum_ += s
        sum_sq_ += sq

    out_idx = b_id * 2
    tl.store(ps3_ptr + out_idx, sum_)
    tl.store(ps3_ptr + out_idx + 1, sum_sq_)


@triton.jit
def layernorm_kernel(
    x_ptr,         # float32 ptr to input x,  shape=(B, LN)
    y_ptr,         # float32 ptr to output y, shape=(B, LN)
    ps3_ptr,       # float32 ptr to final sums array, shape=(B, 2)
    gamma_ptr,     # float32 ptr to gamma of shape=(LN,)
    beta_ptr,      # float32 ptr to beta  of shape=(LN,)
    BLOCK_SIZE: tl.constexpr,
    LN: tl.constexpr,
    B: tl.constexpr,
    EPS: tl.constexpr
):
    """
    Finishes the LN computation in a pass over chunks. For each (b_id, chunk_id),
    read final sums -> mean, var -> read chunk of x -> apply LN -> store y.
    Grid shape is (B, LN//BLOCK_SIZE).
    """
    b_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load sums
    base_idx = b_id * 2
    sum_ = tl.load(ps3_ptr + base_idx)
    sum_sq_ = tl.load(ps3_ptr + base_idx + 1)
    mean_ = sum_ / float(LN)
    var_ = sum_sq_ / float(LN) - mean_ * mean_
    var_ = tl.maximum(var_, 0.0)
    inv_std = 1.0 / tl.sqrt(var_ + EPS)

    # Load chunk of x, gamma, beta
    x_data = tl.load(x_ptr + b_id * LN + offsets)
    gamma_data = tl.load(gamma_ptr + offsets)
    beta_data = tl.load(beta_ptr + offsets)

    # Multiply, store
    xhat = (x_data - mean_) * inv_std
    y = xhat * gamma_data + beta_data
    tl.store(y_ptr + b_id * LN + offsets, y)


def triton_layernorm(x, gamma, beta, eps=1e-5):
    """
    x:     float32, shape (B, LN) with B=16 and LN=2^22
    gamma: float32, shape (LN,)
    beta:  float32, shape (LN,)

    We'll do:
      1) partial_sums_1_kernel -> partial sums per chunk
      2) partial_sums_2_kernel -> merges chunk partials in groups of 256
      3) partial_sums_3_kernel -> merges the partial_sums_2 results
      4) layernorm_kernel      -> finalize LN in chunks
    """
    B = x.shape[0]
    LN = x.shape[1]
    BLOCK_SIZE = 1024  # block size for chunk
    GROUP_SIZE = 256   # how many chunk-partials each partial_sums_2 merges

    chunks_per_batch = LN // BLOCK_SIZE
    # Stage 1 partial sums
    ps1_size = B * chunks_per_batch * 2
    ps1 = torch.empty(ps1_size, dtype=torch.float32, device=x.device)

    # Stage 2 partial sums size
    # merges each group of GROUP_SIZE chunk-partials -> (chunks_per_batch // GROUP_SIZE) partial sums per B
    ps2_size = B * (chunks_per_batch // GROUP_SIZE) * 2
    ps2 = torch.empty(ps2_size, dtype=torch.float32, device=x.device)

    # Stage 3 final sums, shape=(B,2)
    ps3 = torch.empty(B * 2, dtype=torch.float32, device=x.device)

    # partial_sums_1
    grid1 = (B, chunks_per_batch)
    partial_sums_1_kernel[grid1](
        x, ps1,
        BLOCK_SIZE=BLOCK_SIZE, LN=LN, B=B
    )

    # partial_sums_2
    grid2 = (B, chunks_per_batch // GROUP_SIZE)
    partial_sums_2_kernel[grid2](
        ps1, ps2,
        BLOCK_SIZE=BLOCK_SIZE, LN=LN, B=B, GROUP_SIZE=GROUP_SIZE
    )

    # partial_sums_3
    grid3 = (B,)
    partial_sums_3_kernel[grid3](
        ps2, ps3,
        BLOCK_SIZE=BLOCK_SIZE, LN=LN, B=B, GROUP_SIZE=GROUP_SIZE
    )

    # layernorm
    y = torch.empty_like(x)
    grid4 = (B, chunks_per_batch)
    layernorm_kernel[grid4](
        x, y, ps3,
        gamma, beta,
        BLOCK_SIZE=BLOCK_SIZE, LN=LN, B=B,
        EPS=eps
    )
    return y


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization (64x256x256) on a
    batch of 16 using faster Triton kernels with hierarchical reduction.
    """
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        # We store the same parameters as PyTorch LayerNorm
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (16, 64, 256, 256). Flatten last 3 dims => LN=2^22
        B = x.shape[0]  # 16
        LN = x.shape[1] * x.shape[2] * x.shape[3]  # 64*256*256=4194304
        # Flatten input and parameters
        x_flat = x.reshape(B, LN).to(torch.float32).contiguous()
        gamma_flat = self.ln.weight.reshape(LN).to(torch.float32).contiguous()
        beta_flat = self.ln.bias.reshape(LN).to(torch.float32).contiguous()

        # Run our custom Triton LN
        y_flat = triton_layernorm(x_flat, gamma_flat, beta_flat, self.ln.eps)

        # Reshape back and return
        return y_flat.reshape_as(x)
