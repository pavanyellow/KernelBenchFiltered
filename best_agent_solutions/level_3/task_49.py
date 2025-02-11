# level 3 index 49 agent name: KernelAgent Claude 3.5 Sonnet speedup: 6.64x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from einops import rearrange

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized CUDA kernel for segsum and exp operations
template<int BLOCK_SIZE>
__global__ void segsum_exp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int n_heads,
    const int num_chunks,
    const int block_len
) {
    extern __shared__ float shared_mem[];
    
    // Calculate indices
    const int batch_idx = blockIdx.x / (n_heads * num_chunks);
    const int head_idx = (blockIdx.x / num_chunks) % n_heads;
    const int chunk_idx = blockIdx.x % num_chunks;
    const int thread_idx = threadIdx.x;
    
    // Calculate input offset
    const int input_offset = (((batch_idx * n_heads + head_idx) * num_chunks + chunk_idx) * block_len);
    const float* input_block = input + input_offset;
    
    // Load input to shared memory
    if (thread_idx < block_len) {
        shared_mem[thread_idx] = input_block[thread_idx];
    }
    __syncthreads();
    
    // Calculate output indices
    const int output_offset = (((batch_idx * n_heads + head_idx) * num_chunks + chunk_idx) * block_len * block_len);
    float* output_block = output + output_offset;
    
    // Each thread computes multiple elements
    const int elements_per_thread = (block_len * block_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int tid = thread_idx + i * BLOCK_SIZE;
        if (tid < block_len * block_len) {
            const int row = tid / block_len;
            const int col = tid % block_len;
            
            if (row >= col) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = col; k <= row; k++) {
                    sum += shared_mem[k];
                }
                output_block[tid] = expf(sum);
            } else {
                output_block[tid] = 0.0f;
            }
        }
    }
}

// CUDA kernel for decay state computation
__global__ void decay_states_kernel(
    const float* __restrict__ cumsum,
    float* __restrict__ output,
    const int batch_size,
    const int n_heads,
    const int num_chunks,
    const int block_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * n_heads * num_chunks * block_len;
    
    if (idx < total_elements) {
        const int b = idx / (n_heads * num_chunks * block_len);
        const int h = (idx / (num_chunks * block_len)) % n_heads;
        const int c = (idx / block_len) % num_chunks;
        const int l = idx % block_len;
        
        const float last_cumsum = cumsum[((b * n_heads + h) * num_chunks + c) * block_len + (block_len - 1)];
        const float current_cumsum = cumsum[((b * n_heads + h) * num_chunks + c) * block_len + l];
        
        output[idx] = expf(last_cumsum - current_cumsum);
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int n_heads = input.size(1);
    const int num_chunks = input.size(2);
    const int block_len = input.size(3);
    
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    
    auto output = torch::empty({batch_size, n_heads, num_chunks, block_len, block_len}, options);
    
    const int threads = 256;
    const int blocks = batch_size * n_heads * num_chunks;
    const int shared_mem_size = block_len * sizeof(float);
    
    segsum_exp_kernel<256><<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        n_heads,
        num_chunks,
        block_len
    );
    
    return output;
}

torch::Tensor compute_decay_states_cuda(torch::Tensor cumsum) {
    const int batch_size = cumsum.size(0);
    const int n_heads = cumsum.size(1);
    const int num_chunks = cumsum.size(2);
    const int block_len = cumsum.size(3);
    
    auto options = torch::TensorOptions()
        .dtype(cumsum.dtype())
        .device(cumsum.device());
    
    auto output = torch::empty_like(cumsum);
    
    const int threads = 256;
    const int total_elements = batch_size * n_heads * num_chunks * block_len;
    const int blocks = (total_elements + threads - 1) / threads;
    
    decay_states_kernel<<<blocks, threads>>>(
        cumsum.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        n_heads,
        num_chunks,
        block_len
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor segsum_exp_cuda(torch::Tensor input);
torch::Tensor compute_decay_states_cuda(torch::Tensor cumsum);
"""

# Compile the CUDA kernels
cuda_module = load_inline(
    name='mamba_kernels',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['segsum_exp_cuda', 'compute_decay_states_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
    
    def forward(self, X, initial_states=None):
        # Rearrange into blocks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        # Rearrange A_blocks for computation
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # Use optimized CUDA kernel for segsum and exp
        L = cuda_module.segsum_exp_cuda(A_blocks)
        
        # Compute diagonal block outputs using einsum (handled by cuBLAS)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # Use optimized CUDA kernel for decay states
        decay_states = cuda_module.compute_decay_states_cuda(A_cumsum)
        
        # Compute states
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # Handle initial states
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # Compute final decay chunk and states
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        
        return new_states[:, -1]

    def segsum(self, x):
        """Fallback segsum for the final computation"""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
