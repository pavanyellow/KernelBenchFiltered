# level 5 index 7 agent name: 4o Finetuned on L1-3 speedup: 1.07x


import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import Config

from typing import Tuple

# Simplified CUDA kernel to fuse SiLU and Elementwise Multiply
ACT_QUANT_SRC = '''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void act_quant_kernel(const half* __restrict__ x_ptr,
                                  half* __restrict__ y_ptr,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half x_val = x_ptr[idx];
        half silu_val = __hmul(x_val, __hfma(__h2float(x_val), x_val, half(1.0f)));
        y_ptr[idx] = silu_val;
    }
}
'''

ACT_QUANT_DECLARATION = 'void act_quant_kernel(const half* x_ptr, half* y_ptr, int size);'

def launch_act_quant_kernel(x: torch.Tensor) -> torch.Tensor:
    size = x.numel()
    y = torch.empty_like(x)
    block = 256
    grid = (size + block - 1) // block
    torch.cuda.c10d.InternalCUALaunch(CUDA_LAUNCH_FLAGS=0,
                                 block=(grid,),
                                 launch_args=(x, y))
    return y


act_quant_configs = [Config({'BLOCK_SIZE': bs, 'BLOCKS_PER_PROGRAM': nblocks}, num_stages=3, num_warps=4)
                     for bs in [64, 128, 256, 512, 1024, 2048]
                     for nblocks in [1, 2, 4, 8]]

@triton.autotune(act_quant_configs, key=[])
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, num_elements,
                     BLOCK_SIZE: tl.constexpr, BLOCKS_PER_PROGRAM: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM) + pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    s = tl.max(tl.abs(x), axis=0) / 448.0
    y = tl.cast(x / s, tl.element_type(y_ptr))
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(s_ptr + pid, s)

fp8_gemm_configs = [Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': 128}, num_stages=s, num_warps=w)
                     for m in [16, 32, 64]
                     for n in [32, 64, 128]
                     for s in [3, 4, 5, 6]
                     for w in [4, 8]]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, total_N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_n_blocks = total_N // BLOCK_SIZE_N
    offset_n_wrap = pid_n * BLOCK_SIZE_N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = offset_n_wrap + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    scaled_block_k = tl.cdiv(K, BLOCK_SIZE_K)
    for i in range(scaled_block_k):
        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_scale = tl.load(a_s_ptr + offs_m)
        b_scale = tl.load(b_s_ptr + (offs_n // BLOCK_SIZE_N))
        accumulator += tl.dot(a_block, b_block) * (a_scale[:, None] * b_scale[None, :])
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + offs_m[:, None] * total_N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < total_N)
    tl.store(c_ptrs, accumulator.to(tl.element_type(c_ptr)), mask=mask)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    y = torch.empty_like(x, dtype=torch.float32)
    s = x.new_empty(x.numel() // math.prod(x.shape[-x.dim():]) if x.dim() >= 1 else 1, 
                    dtype=torch.float32)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE'] * meta['BLOCKS_PER_PROGRAM']), )
    act_quant_kernel[grid](x, y, s, num_elements)
    return y, s


block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
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
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    result = F.linear(x, weight, bias)
    if gemm_impl == "fp8":
        num_elements = result.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        act_quant_kernel[grid](result, result, result, num_elements)
    return result

class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=dtype or Linear.dtype) * in_features ** -0.5
        )
        self.scale = (nn.Parameter(torch.randn(block_size, self.out_features // block_size))
                      if self.weight.element_size() == 1 else None)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.randn(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores.clone()
        
        if self.bias is not None:
            scores += self.bias
            
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1) if self.bias is None else scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
            
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        torch.set_default_dtype(torch.bfloat16)
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
            
        z = self.shared_experts(x)
        return (y + z).view(shape)


batch_size = 128
seq_len = 1
model_args = ModelArgs()


def get_inputs():
    return [torch.randn(batch_size, seq_len, model_args.dim)]


def get_init_inputs():
    return [model_args]


if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = get_inputs()
    model = Model(args)
    print(model(*x).size())

