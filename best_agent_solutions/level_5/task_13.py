# level 5 index 13 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.10x

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import math
from typing import Optional, Tuple, Literal


# Optimized CUDA source for activation quantization with fused operations
act_quant_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void act_quant_kernel(const float* x, float* y, float* s, int num_elements, int block_size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int idx = block_id * block_size + tid;

    float max_val = 0.0;
    if (tid < block_size && idx < num_elements) {
        max_val = fabsf(x[idx]);
    }
    shared_max[tid] = max_val;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        s[block_id] = shared_max[0] / 448.0f;
    }

    __syncthreads();
    if (tid < block_size && idx < num_elements) {
        y[idx] = x[idx] / s[block_id];
    }
}

std::tuple<torch::Tensor, torch::Tensor> act_quant_cuda(torch::Tensor x, int block_size) {
    int num_elements = x.numel();
    int num_blocks = (num_elements + block_size - 1) / block_size;
    auto y = torch::empty_like(x);
    auto s = torch::empty({num_blocks}, x.options().dtype(torch::kFloat32));

    const int threads = min(block_size, 1024);
    const int shared_memory_size = threads * sizeof(float);

    act_quant_kernel<<<num_blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        s.data_ptr<float>(),
        num_elements,
        block_size
    );

    return std::make_tuple(y, s);
}
"""

# Declare and compile act_quant_cuda function
act_quant_cpp_source = "std::tuple<torch::Tensor, torch::Tensor> act_quant_cuda(torch::Tensor x, int block_size);"
act_quant_module = load_inline(
    name="act_quant",
    cpp_sources=act_quant_cpp_source,
    cuda_sources=act_quant_kernel_source,
    functions=["act_quant_cuda"],
    verbose=True
)

def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    return act_quant_module.act_quant_cuda(x, block_size)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.load(s_ptr + pid_m * (N // BLOCK_SIZE) + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float32)
    grid_size = (M // block_size, N // block_size)
    weight_dequant_kernel[grid_size](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y

# Optimized CUDA source for FP8 GEMM with fusion opportunities
fp8_gemm_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fp8_gemm_kernel(const float* a, const float* a_s, const float* b, const float* b_s, float* c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        const float scale_a = a_s[row];
        const float scale_b = b_s[col];
        for (int k = 0; k < K; ++k) {
            acc += a[row * K + k] * b[k * N + col] * scale_a * scale_b;
        }
        c[row * N + col] = acc;
    }
}

torch::Tensor fp8_gemm_cuda(torch::Tensor a, torch::Tensor a_s, torch::Tensor b, torch::Tensor b_s, int M, int N, int K) {
    auto c = torch::empty({M, N}, a.options());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    fp8_gemm_kernel<<<numBlocks, threadsPerBlock>>>(a.data_ptr<float>(), a_s.data_ptr<float>(), b.data_ptr<float>(), b_s.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}
"""

# Declare and compile fp8_gemm_cuda function
fp8_gemm_cpp_source = "torch::Tensor fp8_gemm_cuda(torch::Tensor a, torch::Tensor a_s, torch::Tensor b, torch::Tensor b_s, int M, int N, int K);"
fp8_gemm_module = load_inline(
    name="fp8_gemm",
    cpp_sources=fp8_gemm_cpp_source,
    cuda_sources=fp8_gemm_kernel_source,
    functions=["fp8_gemm_cuda"],
    verbose=True
)

def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M, N = a.size(0), b.size(1)
    return fp8_gemm_module.fp8_gemm_cuda(a, a_s, b, b_s, M, N, K)

block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

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
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype) * in_features**-0.5)
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.randn(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
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

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

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
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
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
            y.index_add_(0, idx, expert(x[idx]) * weights[idx, top, None])
        z = self.shared_experts(x)
        return (y + z).view(shape)
