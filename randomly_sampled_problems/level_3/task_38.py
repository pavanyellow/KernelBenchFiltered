# level 3 index 38 agent name: KernelAgent o1 speedup: 1.16x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###############################################################################
# In this example, we leave the LSTM exactly as-is (so it still uses cuDNN under 
# the hood for speed), and show a small fused CUDA kernel for adding the FC bias. 
# Normally, this saves only one tiny kernel launch compared to PyTorch’s built-in 
# bias addition. It does demonstrate how to inline custom CUDA code in PyTorch,
# while preserving the exact same forward results (within floating tolerance).
#
# As discussed in the data-dependency analysis, the LSTM itself is already 
# heavily optimized internally, so rewriting or fusing it yields little benefit 
# unless you undertake a complete custom LSTM kernel. Likewise, the final 
# slicing plus matmul is not purely elementwise, so here we only fuse the bias 
# addition as an example.
###############################################################################

# A small inline CUDA kernel that adds a per-column bias to a 2D tensor.
#   x:  (rows, cols) tensor
#   bias: (cols,) tensor
# for each element x[r, c], do x[r, c] += bias[c].

add_bias_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* x, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        x[idx] += bias[c];
    }
}

torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    // Ensure inputs are contiguous
    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    int rows = x_contig.size(0);
    int cols = x_contig.size(1);
    int size = rows * cols;

    // Launch 1D grid of 1D blocks
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    add_bias_kernel<<<grid_size, block_size>>>(x_contig.data_ptr<float>(),
                                              bias_contig.data_ptr<float>(),
                                              rows, cols);

    // The kernel updates x_contig in-place. Return that.
    return x_contig;
}
""".strip()

add_bias_cpp_source = r"torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the inline CUDA code for the bias-add operation
add_bias_module = load_inline(
    name="add_bias_inline_module",
    cpp_sources=add_bias_cpp_source,
    cuda_sources=add_bias_source,
    functions=["add_bias_cuda"],
    verbose=False,
)

###############################################################################
# The same Model interface as before, with:
#   self.h0, self.c0  as initial states
#   self.lstm         as a cuDNN-based LSTM
#   self.fc           as the final Linear layer
# We only replace the final step out = self.fc(...) with a manual matmul plus 
# fused bias addition in a custom kernel.
###############################################################################

batch_size = 10  # As implied by the original code

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        # Keep the exact same random parameter initialization:
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers * 2, batch_size, hidden_size))

        # The LSTM (bidirectional => 2 * hidden_size output)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # A final linear layer from 2*hidden_size => output_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Move initial states to the device of x
        self.h0 = self.h0.to(x.device)
        # (the original code reassigns c0 to h0.to(...) which is presumably a bug, 
        #  but we keep that behavior to match the original exactly)
        self.c0 = self.h0.to(x.device)

        # Let cuDNN handle the main LSTM:
        out, _ = self.lstm(x, (self.h0, self.c0))
        # Slice out the last time-step
        out_last = out[:, -1, :]    # shape [batch_size, 2*hidden_size]

        # Replace self.fc(out_last) with a two-step:
        #   (1) a matmul with self.fc.weight^T
        #   (2) a fused kernel that adds self.fc.bias
        out_matmul = torch.matmul(out_last, self.fc.weight.t())
        out_matmul = add_bias_module.add_bias_cuda(out_matmul, self.fc.bias)

        return out_matmul
