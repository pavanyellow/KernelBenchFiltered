# level 2 index 76 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.33x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_bias_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    const int rows,
    const int cols
) {
    // Use shared memory for bias
    extern __shared__ char shared_mem[];
    scalar_t* shared_bias = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Load bias into shared memory
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();
    
    // Calculate global position
    const int row = blockIdx.x;
    
    if (row < rows) {
        // Process elements
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            scalar_t val = input[row * cols + col] + shared_bias[col];
            output[row * cols + col] = val > 0 ? val : 0;
        }
    }
}

torch::Tensor fused_bias_relu_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    auto output = torch::empty_like(input);
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    const int threads_per_block = 256;
    const int blocks = rows;
    
    // Calculate shared memory size
    const int shared_mem_size = cols * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_bias_relu_kernel", ([&] {
        fused_bias_relu_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            rows,
            cols
        );
    }));
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias);
"""

fused_bias_relu_module = load_inline(
    name='fused_bias_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_bias_relu_cuda'],
    verbose=True
)

class Model(nn.Module):
    """
    Optimized model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    Uses a custom CUDA kernel to fuse bias addition and ReLU operations.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = torch.matmul(x, self.gemm.weight.t())
        x = fused_bias_relu_module.fused_bias_relu_cuda(x, self.bias)
        return x
