# level 3 index 2 agent name: KernelAgent 4o speedup: 1.03x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU
relu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);  // Use empty_like for performance.
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to launch CUDA kernel for ReLU: " + std::string(cudaGetErrorString(err)));
    }

    return out;
}
"""

relu_kernel_cpp_source = "torch::Tensor relu_forward_cuda(torch::Tensor x);"

# Compile the CUDA extension for the custom ReLU
relu_native_module = load_inline(
    name='relu_native',
    cpp_sources=relu_kernel_cpp_source,
    cuda_sources=relu_kernel_source,
    functions=['relu_forward_cuda'],
    verbose=True
)

# Define a custom layer with the CUDA ReLU function
class CustomReLU(nn.Module):
    def forward(self, x):
        return relu_native_module.relu_forward_cuda(x)

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(CustomReLU())  # Use CustomReLU instead of nn.ReLU
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size, device='cuda')]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]

# Set up for testing
if __name__ == "__main__":
    inputs = get_inputs()
    model = Model(*get_init_inputs()).to('cuda')
    output = model(inputs[0])
    print(output)
