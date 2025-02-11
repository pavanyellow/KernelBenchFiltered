# level 1 index 16 agent name: KernelAgent 4o speedup: 1.19x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define the Triton kernel for matrix transposition
@triton.jit
def transpose_kernel(a_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Indices for accessing the elements
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load a data block from A, assume contiguous input and power of two dimension
    a = tl.load(a_ptr + (offs_m[:, None] * N + offs_n[None, :]))

    # Store transposed block to output
    tl.store(out_ptr + (offs_n[:, None] * M + offs_m[None, :]), a.T)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Create an output tensor for the transpose result
        A_transposed = torch.empty((A.size(1), A.size(0)), device=A.device, dtype=A.dtype)

        # Execute the Triton kernel to transpose A
        BLOCK_SIZE = 64  # Choosing 64 to improve SM occupancy and memory throughput
        grid = (triton.cdiv(A.size(0), BLOCK_SIZE), triton.cdiv(A.size(1), BLOCK_SIZE))
        transpose_kernel[grid](A, A_transposed, A.size(0), A.size(1), BLOCK_SIZE=BLOCK_SIZE)

        # Utilize high-performance torch.matmul which uses cuBLAS for optimal performance
        return torch.matmul(A_transposed, B)

# Example usage:
# A = torch.randn(4096, 1024, device='cuda', dtype=torch.float32)
# B = torch.randn(4096, 2048, device='cuda', dtype=torch.float32)
# model = Model()
# C = model(A, B)
