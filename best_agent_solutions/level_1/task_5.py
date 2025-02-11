# level 1 index 5 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.01x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matscalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scalar,
    const int M,
    const int N
) {
    // Use shared memory for better memory coalescing
    __shared__ float4 shared_mem[32][32];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    
    // Global indices
    const int row = bid_y * blockDim.y + tid_y;
    const int col = bid_x * blockDim.x * 4 + tid_x * 4;
    
    if (row < M && col < N) {
        // Load 4 elements at once using vectorized load
        float4* in_ptr = (float4*)&input[row * N + col];
        float4 val = *in_ptr;
        
        // Multiply by scalar
        val.x *= scalar;
        val.y *= scalar;
        val.z *= scalar;
        val.w *= scalar;
        
        // Store directly to output
        float4* out_ptr = (float4*)&output[row * N + col];
        *out_ptr = val;
    }
    
    // Handle edge cases where N is not divisible by 4
    const int remainder = N % 4;
    if (remainder > 0 && row < M && col == (N - remainder)) {
        for (int i = 0; i < remainder; i++) {
            output[row * N + col + i] = input[row * N + col + i] * scalar;
        }
    }
}

torch::Tensor matscalar_cuda(torch::Tensor input, float scalar) {
    auto output = torch::empty_like(input);
    
    const int M = input.size(0);
    const int N = input.size(1);
    
    // Configure kernel launch parameters
    dim3 threads_per_block(32, 32);  // 32x32 thread block
    dim3 num_blocks(
        (N + (threads_per_block.x * 4) - 1) / (threads_per_block.x * 4),
        (M + threads_per_block.y - 1) / threads_per_block.y
    );
    
    matscalar_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar,
        M,
        N
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor matscalar_cuda(torch::Tensor input, float scalar);
"""

matscalar_module = load_inline(
    name='matscalar_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matscalar_cuda'],
    verbose=True
)

class Model(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        if not A.is_cuda:
            A = A.cuda()
        if not A.is_contiguous():
            A = A.contiguous()
        
        return matscalar_module.matscalar_cuda(A, s)

# Keep the same test parameters
M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []
