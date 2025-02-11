# level 3 index 1 agent name: KernelAgent 4o speedup: 1.02x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise ReLU activation
relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(in[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor in);"

# Compile the inline CUDA code for ReLU activation
relu_native_module = load_inline(
    name='relu_cuda',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=['relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
            
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.linears = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        for layer in self.linears[:-1]:
            x = layer(x)  # Efficient matrix multiplication with PyTorch
            x = relu_native_module.relu_cuda(x)  # Custom CUDA ReLU
        x = self.linears[-1](x)  # The final linear layer
        return x

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
