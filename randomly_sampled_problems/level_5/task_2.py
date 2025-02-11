# level 5 index 2 agent name: KernelAgent O3 Mini High speedup: 1.29x

import math
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

# Import Triton and its language module for our fused attention kernel.
import triton
import triton.language as tl

#################################################################################################
#               Fused Attention Kernel via Triton with Autotuning Parameters                   #
#################################################################################################

# Default values (used if autotuning is disabled) – these will be overridden by autotune.
_DEFAULT_BLOCK_M = 16   # Default tile size for query tokens (must divide L)
_DEFAULT_BLOCK_N = 16   # Default tile size for key tokens
_DEFAULT_BLOCKS_PER_PROG = 1  # Default number of key–tiles processed per kernel program

# Define a list of autotune candidate configurations.
# We provide ~5 candidate block sizes (i.e. BLOCK_M * BLOCK_N roughly between 64 and 2048 total elements)
# and allow the kernel to loop over multiple blocks per program, with BLOCKS_PER_PROG in {1, 2, 4, 8}.
# Also, we try num_warps set to either 4 or 8.
autotune_configs = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 8, 'BLOCKS_PER_PROG': 1}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 8, 'BLOCKS_PER_PROG': 4}, num_stages=1, num_warps=8),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCKS_PER_PROG': 2}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCKS_PER_PROG': 2}, num_stages=1, num_warps=8),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCKS_PER_PROG': 2}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCKS_PER_PROG': 4}, num_stages=1, num_warps=8),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCKS_PER_PROG': 1}, num_stages=1, num_warps=4),
]

@triton.autotune(configs=autotune_configs, key=['L', 'head_dim', 'M'])
@triton.jit
def fused_attention_kernel(Q, K, V, O,
                           L: tl.constexpr, head_dim: tl.constexpr, M: tl.constexpr,  # L and head_dim are compile-time; M is total batch–head slices
                           # Since inputs are contiguous, we assume 0–indexed strides.
                           stride_q: tl.constexpr, stride_k: tl.constexpr, stride_v: tl.constexpr, stride_o: tl.constexpr,
                           scale: tl.constexpr, 
                           BLOCK_M: tl.constexpr,     # tile size for queries; L must be divisible by BLOCK_M
                           BLOCK_N: tl.constexpr,     # tile size for keys (per iteration)
                           BLOCKS_PER_PROG: tl.constexpr  # number of key–tiles (of size BLOCK_N) processed per program
                           ):
    # Total number of query tiles per batch–head slice.
    n_qtiles = L // BLOCK_M

    # Compute which batch–head slice and which query–tile this kernel instance will process.
    pid = tl.program_id(0)
    bh = pid % M         # which batch–head slice (0 <= bh < M)
    tile_id = pid // M   # which query–tile (0 <= tile_id < n_qtiles)
    q_start = tile_id * BLOCK_M  # starting index (in the query sequence) for this tile

    # Precompute query offsets and the column indices (for head_dim)
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, head_dim)

    # Pointer to the beginning of the bh–th Q slice and load the query tile (no masking as L is a multiple).
    q_ptr = Q + bh * L * stride_q
    q_tile = tl.load(q_ptr + q_offsets[:, None] * stride_q + col_idx[None, :])

    # Initialize numerical stability accumulators.
    max_val = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    # Pointers for K and V for this batch–head slice.
    k_ptr = K + bh * L * stride_k
    v_ptr = V + bh * L * stride_v

    # The total number of key–tiles processed by this kernel instance is:
    # total_key_tiles = L / BLOCK_N.
    # We process BLOCKS_PER_PROG key–tiles per outer loop iteration.
    n_iters = L // (BLOCK_N * BLOCKS_PER_PROG)
    for i in range(n_iters):
        for j in range(BLOCKS_PER_PROG):
            # Compute current key offset.
            offset = (i * BLOCKS_PER_PROG + j) * BLOCK_N
            k_offsets = offset + tl.arange(0, BLOCK_N)
            # Load key and value tiles (no masks needed).
            k_tile = tl.load(k_ptr + k_offsets[:, None] * stride_k + col_idx[None, :])
            v_tile = tl.load(v_ptr + k_offsets[:, None] * stride_v + col_idx[None, :])
            # Compute scaled dot–product between query tile and current key tile.
            scores = tl.dot(q_tile, k_tile, trans_b=True) * scale  # shape: (BLOCK_M, BLOCK_N)
            block_max = tl.max(scores, axis=1)  # (BLOCK_M,)
            exp_scores = tl.exp(scores - block_max[:, None])
            block_sum = tl.sum(exp_scores, axis=1)
            new_max = tl.maximum(max_val, block_max)
            # Numerically stable update of accumulator and sum_exp.
            acc = acc * tl.exp(max_val - new_max) + tl.dot(exp_scores, v_tile) * tl.exp(block_max - new_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + block_sum * tl.exp(block_max - new_max)
            max_val = new_max
    out_tile = acc / sum_exp[:, None]
    o_ptr = O + bh * L * stride_o
    tl.store(o_ptr + q_offsets[:, None] * stride_o + col_idx[None, :], out_tile)

# Python wrapper mimicking torch.nn.functional.scaled_dot_product_attention.
# Expects Q, K, V of shape (B, L, embed_dim) and a heads argument.
# Internally they are reshaped to (B, heads, L, head_dim), flattened, and passed to the fused kernel.
def triton_attention(q, k, v, heads):
    B, L, D = q.shape
    head_dim = D // heads
    # Reshape from (B, L, embed_dim) to (B, heads, L, head_dim) then transpose and flatten the first two dimensions.
    q = q.view(B, L, heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(B, L, heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(B, L, heads, head_dim).transpose(1, 2).contiguous()
    orig_shape = q.shape  # (B, heads, L, head_dim)
    q = q.reshape(-1, L, head_dim)
    k = k.reshape(-1, L, head_dim)
    v = v.reshape(-1, L, head_dim)
    M_total = q.shape[0]  # total number of batch–head slices
    o = torch.empty_like(q)
    # Grid computation: each kernel instance handles one query tile.
    grid = lambda meta: (M_total * (L // meta['BLOCK_M']),)
    fused_attention_kernel[grid](
        q, k, v, o,
        L, head_dim, M_total,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        1.0 / math.sqrt(head_dim),
        _DEFAULT_BLOCK_M, _DEFAULT_BLOCK_N, _DEFAULT_BLOCKS_PER_PROG
    )
    # Reshape back to (B, heads, L, head_dim) then merge heads.
    o = o.reshape(B, heads, L, head_dim).transpose(1, 2).reshape(B, L, heads * head_dim)
    return o

#################################################################################################
###                               Core/Utility Functions                                    ###
#################################################################################################

# Basic attention wrapper around torch's built-in scaled_dot_product_attention.
def attention(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x

def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scaling_factor=None, offset=None):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        factor = -math.log(max_period) / half
        self.register_buffer("freqs", torch.exp(torch.arange(0, half, dtype=torch.float32, device=device) * factor))

    def timestep_embedding(self, t):
        args = t[:, None].float() * self.freqs.to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""
    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]

def optimized_attention(qkv, num_heads):
    return attention(qkv[0], qkv[1], qkv[2], num_heads)

class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug", "triton")
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_mode: str = "triton",
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        assert attn_mode in SelfAttention.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only
        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)
    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv_out = self.qkv(x)
        q, k, v = split_qkv(qkv_out, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)
    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        if self.attn_mode == "triton":
            out = triton_attention(q, k, v, self.num_heads)
        else:
            out = attention(q, k, v, self.num_heads)
        out = self.post_attention(out)
        return out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x

class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug", "triton")
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "triton",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm,
                                  rmsnorm=rmsnorm, dtype=dtype, device=device)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device)
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only
    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None
    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads) if self.attn.attn_mode != "triton" else triton_attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)

def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)
    x_qkv, x_intermediates = x_block.pre_attention(x, c)
    o = [torch.cat((context_qkv[t], x_qkv[t]), dim=1) for t in range(3)]
    q, k, v = tuple(o)
    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:]
    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x

class JointBlock(nn.Module):
    """Just a small wrapper to serve as a FSDP unit."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)
    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None, dtype=None, device=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        if total_out_channels is None:
            self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        else:
            self.linear = nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Model(nn.Module):
    """Diffusion model with a Transformer backbone."""
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 4,
        depth: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        register_length: int = 0,
        attn_mode: str = "triton",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = 64,
        num_patches = 4096,  # (number of patches)
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        dtype = None,
        device = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        # Hidden size is set to 64 * depth.
        hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True,
                                      strict_img_size=(self.pos_embed_max_size is None), dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device)
        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(**context_embedder_config["params"], dtype=dtype, device=device)
        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode,
                           pre_only=(i == depth - 1), rmsnorm=rmsnorm, scale_mod_only=scale_mod_only,
                           swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device)
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

        self.internal_dtype = torch.float16
        self.half()

    # Optimized cropped_pos_embed using reshape and slicing.
    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top: top + h, left: left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    # Optimized unpatchify using view and permute.
    def unpatchify(self, x, hw=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]
        x = x.view(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward_core_with_concat(self, x: torch.Tensor, c_mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.register_length > 0:
            context = torch.cat(
                (torch.repeat_interleave(self.register, x.shape[0], dim=0)
                 if self.register.shape[0] == 1 else self.register,
                 context if context is not None else torch.empty(0, device=x.device, dtype=x.dtype)),
                1,
            )
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)
        x = self.final_layer(x, c_mod)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(self.internal_dtype)
        t = t.to(self.internal_dtype)
        if y is not None:
            y = y.to(self.internal_dtype)
        if context is not None:
            context = context.to(self.internal_dtype)
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=self.internal_dtype)
        if y is not None:
            y = self.y_embedder(y)
            c = c + y
        context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context)
        x = self.unpatchify(x, hw=hw)
        return x.to(orig_dtype)

#############################
# Test routines and example #
#############################

batch_size = 8
C = 4
H = 256
W = 256

def get_inputs():
    x = torch.randn(batch_size, C, H, W)
    t = torch.full((batch_size,), 3.0)
    context = torch.randn(batch_size, 64 * 28)
    return [x, t, None, context]

def get_init_inputs():
    return []

if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = Model()
    inputs = get_inputs()
    print(model(*inputs).shape)
