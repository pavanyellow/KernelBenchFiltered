# level 5 index 12 agent name: KernelAgent O3 Mini High speedup: 1.01x

import torch
import torch.nn as nn
import math
from typing import Optional, Literal
from dataclasses import dataclass
from torch.nn import Linear, RMSNorm

@dataclass
class ModelArgs:
    max_batch_size: int = 128
    max_seq_len: int = 4096 + 1
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 8192
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    
class Model(nn.Module):
    def __init__(self, args: ModelArgs, *precomputed_input_args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)
        
        # Combine multiplications in a single step.
        base_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = base_scale * (mscale ** 2)
        else:
            self.softmax_scale = base_scale

        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        
        # Precompute fixed prompt using the provided extra arguments.
        self(*precomputed_input_args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        q = self.wq_a(x)
        q = self.q_norm(q)
        q = self.wq_b(q)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # unsqueeze to add a head dimension before rotary.
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        kv = self.kv_norm(kv)
        kv = self.wkv_b(kv)
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            self.k_cache[:bsz, :end_pos],
            self.v_cache[:bsz, :end_pos],
            scale=self.softmax_scale,
            is_causal=False
        )
        out = self.wo(out.flatten(2))
        return out

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Convert x (in pairs of last dimension) to complex, multiply by freqs_cis,
    and return to the original dtype.
    """
    dtype = x.dtype
    # If x is already fp32, x.to(torch.float32) is a no-op.
    x = torch.view_as_complex(x.to(torch.float32).view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precompute rotary frequency values.
    Applies an algebraic simplification so that the multiplication
    of freqs by a factor is computed as:
      smooth*(1 - 1/factor) + 1/factor
    instead of multiplying twice.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(args.beta_fast, args.beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        inv_factor = 1.0 / factor
        # Algebraically simplified: combine the two multiplications into one.
        freqs = freqs * (smooth * (1 - inv_factor) + inv_factor)
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# Testing code remains unchanged.
batch_size = 2
seq_len = 1
precomputed_seq_len = 4096
model_args = ModelArgs()
freqs_cis = precompute_freqs_cis(model_args).to('cuda')

def get_inputs():
    return [torch.randn(batch_size, seq_len, model_args.dim), precomputed_seq_len, freqs_cis[:seq_len]]

def get_precomputed_inputs():
    return [torch.randn(batch_size, precomputed_seq_len, model_args.dim), 0, freqs_cis[:precomputed_seq_len]]

def get_init_inputs():
    return [model_args, *get_precomputed_inputs()]

if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    print(model(*inputs).shape)
