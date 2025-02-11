# level 5 index 14 agent name: 4o Finetuned on L1-3 speedup: 1.04x


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 8
    n_heads: int = 16
    n_kv_heads: int = None
    vocab_size: int = 2048
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 4097

# -----------------------------------------------------------------------------
# CUDA source for RMSNorm: processes every block of size 'dim'
# -----------------------------------------------------------------------------

rmsnorm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each block processes one 'row' of the tensor
__global__ void rmsnorm_kernel(float* x, const float* weight, float eps, int dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int offset = row * dim;
    float sum = 0.0f;
    float inv_dim = 1.0f / float(dim);

    // Each thread processes a portion of the row
    if (tid < dim) {
        float val = x[offset + tid];
        sum = val * val;
    }
    for (int d = tid + blockDim.x; d < dim; d += blockDim.x) {
        float val = x[offset + d];
        sum += val * val;
    }

    // Reduce within block using shared memory
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] * inv_dim + eps);

    // Apply normalization
    if (tid < dim) {
        x[offset + tid] *= scale * weight[tid];
    }
    for (int d = tid + blockDim.x; d < dim; d += blockDim.x) {
        x[offset + d] *= scale * weight[d];
    }
}

void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, float eps) {
    int dim = x.size(-1);
    int n = x.numel() / dim;  // total number of rows
    int threads = min(dim, 256);
    rmsnorm_kernel<<< n, threads, threads * sizeof(float) >>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), eps, dim);
}
"""

rmsnorm_cpp_source = r"void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, float eps);"

rmsnorm_native_module = load_inline(
    name='rmsnorm',
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_cuda_source,
    functions=['rmsnorm_cuda'],
    verbose=True
)

# -----------------------------------------------------------------------------
# Helper functions remain identical
# -----------------------------------------------------------------------------

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    assert head_dim % 2 == 0
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta_values = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len)
    freqs = torch.outer(m.float(), theta_values)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    reshaped_freqs = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * reshaped_freqs
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        batch_size, seq_len, n_kv_heads, n_rep, head_dim
    ).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)


# -----------------------------------------------------------------------------
# Module classes with an optimized RMSNorm
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.register_buffer(
            "cache_k",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        )
        self.register_buffer(
            "cache_v",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish_x = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish_x * x_V
        x = self.w2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # Use the custom CUDA implementation
    def forward(self, x: torch.Tensor):
        if not x.is_cuda:
            x = x.cuda()
        x = x.clone()  # We normalize in-place
        rmsnorm_native_module.rmsnorm_cuda(x, self.weight, self.eps)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Model(nn.Module):
    def __init__(self, args: ModelArgs, *precomputed_input_args):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.register_buffer(
            "freqs_complex",
            precompute_theta_pos_frequencies(
                self.args.dim // self.args.n_heads,
                self.args.max_seq_len * 2,
            )
        )
        
        self(*precomputed_input_args)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output


