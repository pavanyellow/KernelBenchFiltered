# level 2 index 56 agent name: KernelAgent 4o speedup: 2.21x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the optimized CUDA kernel with better utilization
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to perform linear transformation, sigmoid activation, and sum reduction
__global__ void fused_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int input_size, 
    int hidden_size) {

    int row = blockIdx.x;
    int col = threadIdx.x;

    extern __shared__ float shared_data[];

    if (col < hidden_size) {
        float z = bias[col];
        for (int i = 0; i < input_size; ++i) {
            z += x[row * input_size + i] * weight[col * input_size + i];
        }
        shared_data[col] = 1.0f / (1.0f + expf(-z));
    }

    __syncthreads();

    if (col == 0) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += shared_data[i];
        }
        output[row] = sum;
    }
}

// Function to call the fused kernel
torch::Tensor fused_operation(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int hidden_size) {
    auto num_rows = x.size(0);
    auto output = torch::empty({num_rows, 1}, x.options());

    const int threads_per_block = hidden_size;  // Compute optimal number of threads
    const int shared_memory_size = hidden_size * sizeof(float);

    fused_kernel<<<num_rows, threads_per_block, shared_memory_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), x.size(1), hidden_size);

    return output;
}
"""

cpp_source = """
torch::Tensor fused_operation(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int hidden_size);
"""

# Load the inline CUDA kernel
optimized_model_module = load_inline(
    name='fused_model_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_operation'],
    extra_cuda_cflags=['--expt-relaxed-constexpr'],
    verbose=True
)

# Define the Model class using the optimized kernels
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # Extract weight and bias for linear transformation
        weight = self.linear.weight
        bias = self.linear.bias

        # Use the fused custom CUDA kernel
        output = optimized_model_module.fused_operation(x.contiguous(), weight.contiguous(), bias.contiguous(), self.hidden_size)

        return output
