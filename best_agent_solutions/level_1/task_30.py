# level 1 index 30 agent name: KernelAgent 4o speedup: 2.38x

import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous(), "Input tensor must be contiguous"

        # Prepare an empty tensor for the output
        output = torch.empty_like(x)

        # Total number of elements
        N = x.numel()

        # Define grid size to cover all elements
        BLOCK_SIZE = 256
        n_blocks_per_program = 8
        grid = (triton.cdiv(N, BLOCK_SIZE * n_blocks_per_program),)

        # Launch the Triton kernel
        softsign_kernel[grid](x, output, N)

        return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs, 'n_blocks_per_program': np, 'num_warps': nw}, num_stages=1)
        for bs in [64, 128, 256, 512, 1024, 2048]
        for np in [1, 2, 4, 8]
        for nw in [4, 8]
    ],
    key=['x_ptr']
)
@triton.jit
def softsign_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, n_blocks_per_program: tl.constexpr):
    # Calculate index range this kernel should compute
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * n_blocks_per_program + tl.arange(0, BLOCK_SIZE)

    # Iterate over blocks assigned to this program
    for i in range(n_blocks_per_program):
        idx = offsets + i * BLOCK_SIZE
        # Apply mask for valid indices
        mask = idx < N
        
        # Load input data, apply Softsign activation, and store the result
        x = tl.load(x_ptr + idx, mask=mask)
        result = x / (1 + tl.abs(x))
        tl.store(output_ptr + idx, result, mask=mask)
