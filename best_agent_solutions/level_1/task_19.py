# level 1 index 19 agent name: KernelAgent o1 speedup: 1.05x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# We'll define a small custom Triton kernel for ReLU
# that should run faster than a naive Python loop.
# It preserves the same behavior as torch.relu.

@triton.jit
def relu_kernel(InPtr, OutPtr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(InPtr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(OutPtr + offsets, y, mask=mask)

def custom_relu(x: torch.Tensor) -> torch.Tensor:
    # If the input is on CPU, just use torch.relu
    if not x.is_cuda:
        return torch.relu(x)
    # Allocate output
    y = torch.empty_like(x)
    N = x.numel()
    # We'll use a BLOCK_SIZE of 1024 and compute how many blocks we need
    grid = lambda META: ((N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    relu_kernel[grid](x, y, N, BLOCK_SIZE=1024)
    return y

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Same initialization pattern as the original
        # (which had no parameters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace torch.relu with our Triton-accelerated version
        return custom_relu(x)
