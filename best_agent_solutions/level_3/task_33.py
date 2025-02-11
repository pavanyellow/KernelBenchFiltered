# level 3 index 33 agent name: KernelAgent 4o speedup: 1.20x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tanh activation kernel
__global__ void tanh_activation_kernel(const float* __restrict__ data_in, float* data_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data_out[idx] = tanhf(data_in[idx]);
    }
}

// Concatenation kernel for 2 tensors along dim=1
__global__ void cat_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, 
                           int a_cols, int b_cols, int out_cols, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy < rows && idx < (a_cols + b_cols)) {
        int out_index = idy * out_cols + idx;
        if (idx < a_cols) {
            out[out_index] = a[idy * a_cols + idx];
        } else {
            out[out_index] = b[idy * b_cols + (idx - a_cols)];
        }
    }
}

torch::Tensor tanh_activation_cuda(torch::Tensor data) {
    auto size = data.numel();
    auto results = torch::empty_like(data);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    tanh_activation_kernel<<<num_blocks, block_size>>>(data.data_ptr<float>(), results.data_ptr<float>(), size);
    return results;
}

torch::Tensor concatenate_cuda(torch::Tensor a, torch::Tensor b) {
    int rows = a.size(0);
    int a_cols = a.size(1);
    int b_cols = b.size(1);
    int out_cols = a_cols + b_cols;
    auto out = torch::empty({rows, out_cols}, a.options());

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((out_cols + 15) / 16, (rows + 15) / 16);
    cat_kernel<<<num_blocks, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a_cols, b_cols, out_cols, rows);

    return out;
}
"""

cuda_cpp_source = """
torch::Tensor tanh_activation_cuda(torch::Tensor data);
torch::Tensor concatenate_cuda(torch::Tensor a, torch::Tensor b);
"""

cuda_native_module = load_inline(
    name='cuda_operations',
    cpp_sources=cuda_cpp_source,
    cuda_sources=cuda_kernels_source,
    functions=['tanh_activation_cuda', 'concatenate_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_hidden = nn.Parameter(torch.randn((8, hidden_size), requires_grad=False))
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        if initial_hidden is None:
            hidden = self.initial_hidden.to(x.device)
        else:
            hidden = initial_hidden.to(x.device)

        # Use the custom CUDA kernel for concatenation
        combined = cuda_native_module.concatenate_cuda(x, hidden)
        i2h_out = self.i2h(combined)

        # Use the custom CUDA kernel for the Tanh activation
        i2h_out = cuda_native_module.tanh_activation_cuda(i2h_out)

        output = self.h2o(i2h_out)
        return output
