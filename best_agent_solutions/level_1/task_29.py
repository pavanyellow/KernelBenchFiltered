# level 1 index 29 agent name: KernelAgent 4o speedup: 1.09x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define Triton kernel for Softplus
@triton.jit
def softplus_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Iterating over the elements with block size steps
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = tl.program_id(0) * BLOCK_SIZE + offsets
    # Load elements
    x = tl.load(x_ptr + idx, mask=idx < n_elements, other=0.0)
    # Compute softplus
    out = tl.log(1 + tl.exp(x))
    # Write back
    tl.store(output_ptr + idx, out, mask=idx < n_elements)

class Model(nn.Module):
    """
    Simple model that performs a Softplus activation using Triton for optimization.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare output tensor
        output = torch.empty_like(x)
        # Number of elements in the input
        n_elements = x.numel()
        # Launch Triton kernel
        BLOCK_SIZE = 1024  # Play with this number if needed based on your device architecture
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        softplus_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []

