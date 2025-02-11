# level 2 index 51 agent name: KernelAgent 4o speedup: 2.79x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for Subtract, GlobalAvgPool, LogSumExp and GELU
fused_operations_cuda_source = """
#include <torch/extension.h>
#include <cmath>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_operations_kernel(const float* input, const float* subtract, 
                                        float* output, int rows, int cols) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * cols + col;

    if (col < cols) {
        // Subtract
        float val = input[idx] - subtract[col];
        shared_data[col] = val;  // Store for reduction
    }

    __syncthreads();

    // First step: GlobalAvgPool
    if (col < cols) {
        for (unsigned int stride = cols / 2; stride > 0; stride >>= 1) {
            if (col < stride) {
                shared_data[col] += shared_data[col + stride];
            }
            __syncthreads();
        }
    }

    // LogSumExp: We need a numerically stable way
    __syncthreads();
    if (col == 0) {
        float mean = shared_data[0] / static_cast<float>(cols);
        float max_val = mean;
        float sum_exp = expf(mean - max_val);

        shared_data[0] = logf(sum_exp) + max_val;  // Store the log-sum-exp

        // Calculate GELU and update output
        output[row] = gelu(shared_data[0]);
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor subtract) {
    auto output = torch::empty({input.size(0), 1}, input.options());
    int rows = input.size(0);
    int cols = input.size(1);

    int block_size = cols;
    int num_blocks = rows;

    fused_operations_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), subtract.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}
"""

# Compile the CUDA kernels
cuda_kernels = load_inline(
    name='cuda_kernels_optimized_rewrite',
    cpp_sources="""
        torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor subtract);
    """,
    cuda_sources=fused_operations_cuda_source,
    functions=['fused_operations_cuda'],
    verbose=True,
)

class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features, device='cuda'))
    
    def forward(self, x):
        original_x = x.clone().detach()

        # GEMM Operation
        x = self.gemm(x)

        # Fused operations using custom CUDA kernel
        x = cuda_kernels.fused_operations_cuda(x, self.subtract)

        # Residual Add, expand x to match original_x dimensions
        x = x.expand_as(original_x) + original_x

        return x

def get_inputs():
    batch_size = 128
    in_features = 1024
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]
