# level 1 index 21 agent name: KernelAgent 4o speedup: 1.07x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Optimized Triton kernel for the sigmoid function
@triton.jit
def sigmoid_kernel_optimized(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from X, compute sigmoid using Triton's sigmoid function, and store in Y
    x = tl.load(X + offsets)
    y = tl.sigmoid(x)
    tl.store(Y + offsets, y)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous(), "Input tensor must be contiguous"
        N = x.numel()
        y = torch.empty_like(x)

        # Launch Triton kernel
        BLOCK_SIZE = 1024  # Adjust based on the hardware for optimal performance
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        sigmoid_kernel_optimized[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        return y
