# level 3 index 41 agent name: KernelAgent 4o speedup: 1.05x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for initializing the hidden state tensor
hidden_state_init_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_hidden_kernel(float* h0, int size, unsigned long long seed, bool is_zero) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    if (idx < size) {
        if (is_zero) {
            h0[idx] = 0.0f;
        } else {
            h0[idx] = curand_normal(&state);
        }
    }
}

torch::Tensor init_hidden_cuda(int num_layers, int batch_size, int hidden_size, bool is_zero) {
    auto h0 = torch::empty({num_layers, batch_size, hidden_size}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    int size = h0.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    unsigned long long seed = 42;  // Fixed seed for reproducible results

    init_hidden_kernel<<<num_blocks, block_size>>>(h0.data_ptr<float>(), size, seed, is_zero);
    cudaDeviceSynchronize();

    return h0;
}
"""

hidden_state_init_cpp_source = """
torch::Tensor init_hidden_cuda(int num_layers, int batch_size, int hidden_size, bool is_zero);
"""

# Compile the inline CUDA code
init_hidden_native_module = load_inline(
    name='init_hidden',
    cpp_sources=hidden_state_init_cpp_source,
    cuda_sources=hidden_state_init_source,
    functions=['init_hidden_cuda'],
    verbose=False
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional_factor = 2
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)

    def forward(self, x, h0=None):
        batch_size = x.size(1) if not self.gru.batch_first else x.size(0)
        
        if h0 is None:
            h0 = init_hidden_native_module.init_hidden_cuda(
                self.num_layers * self.bidirectional_factor, 
                batch_size, 
                self.hidden_size,
                False  # False for random initialization
            )
        
        output, h_n = self.gru(x, h0)
        return output

# Usage Example
if __name__ == "__main__":
    batch_size = 10
    seq_len = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6

    model = Model(input_size, hidden_size, num_layers).to('cuda')
    
    def get_inputs():
        seq_length = 512
        return [torch.randn(seq_length, batch_size, input_size, device='cuda'), 
                torch.randn((num_layers*2, batch_size, hidden_size), device='cuda')]
    
    inputs = get_inputs()
    output = model(*inputs)
    print("Output shape:", output.shape)  # Expect shape: (seq_len, batch_size, num_directions * hidden_size)
