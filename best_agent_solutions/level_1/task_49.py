# level 1 index 49 agent name: KernelAgent O3 Mini High speedup: 2.51x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# This kernel is fully specialized for inputs of shape (16, 256, 256)
# It reduces along dimension 1 (of length 256) without any looping.
#
# For each output element (corresponding to a unique (b, m) with
# b in [0,16) and m in [0,256)), the result is:
#
#    y[b, m] = max_{d in [0,256)} x[b, d, m]
#
# Memory layout: x is contiguous in row–major order, so the offset for
# a given (b, d, m) is:  b*(256*256) + d*256 + m
#
# Because all dimensions are powers‐of‐2, we can compute all indices with no masks.
#
# We use autotuning only over BLOCK_SIZE (the number of output elements processed
# per kernel instance) and the number of warps.
#
# Note: We make B, D, and M compile‐time constants by adding them as constexpr parameters.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=1),
    ],
    key=[]
)
@triton.jit
def _max_reduction_kernel(x_ptr, out_ptr, 
                          BLOCK_SIZE: tl.constexpr,
                          B: tl.constexpr = 16,      # Batch size (compile-time constant)
                          D: tl.constexpr = 256,     # Reduction dimension (to be maximized over)
                          M: tl.constexpr = 256):    # The remaining dimension for output
    # Total number of elements per batch in x (D*M)
    DM = D * M

    # Each kernel instance processes BLOCK_SIZE output elements.
    # Total number of output elements is B * M = 16 * 256 = 4096.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)      # (BLOCK_SIZE,) -- BLOCK_SIZE is constexpr.
    idx = pid * BLOCK_SIZE + offs        # Flat index in [0, B*M)

    # Compute b and m for each output element:
    # b = idx // M, m = idx % M.
    b_val = idx // M                     # (BLOCK_SIZE,)
    m_val = idx % M                      # (BLOCK_SIZE,)

    # Reshape for broadcasting in later arithmetic.
    b_val = tl.reshape(b_val, (BLOCK_SIZE, 1))  # shape: (BLOCK_SIZE, 1)
    m_val = tl.reshape(m_val, (BLOCK_SIZE, 1))  # shape: (BLOCK_SIZE, 1)

    # Create a vector for the reduction dimension. D is a constexpr.
    d = tl.arange(0, D)                         # shape: (D,)

    # Compute addresses for x[b, d, m]:
    # Address = b * (D*M) + d*M + m, resulting in a (BLOCK_SIZE, D) tensor of indices.
    addrs = b_val * DM + m_val + d * M             # shape: (BLOCK_SIZE, D)

    # Gather-load the D values per output element.
    vals = tl.load(x_ptr + addrs)             # shape: (BLOCK_SIZE, D)

    # Compute the maximum along the reduction axis (d dimension).
    max_val = tl.max(vals, axis=1)            # shape: (BLOCK_SIZE,)

    # Write the reduced results to the output tensor.
    tl.store(out_ptr + idx, max_val)


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If the input is not the specialized (16, 256, 256) reduction along dim=1,
        # fall back to PyTorch's max.
        if self.dim != 1 or x.ndim != 3 or x.shape != (16, 256, 256):
            return torch.max(x, dim=self.dim)[0]

        # For x of shape (16, 256, 256) reducing over dim=1, the output is (16, 256)
        output = torch.empty((16, 256), device=x.device, dtype=x.dtype)
        total_elems = 16 * 256  # 4096 output elements
        grid = lambda META: (triton.cdiv(total_elems, META['BLOCK_SIZE']),)
        _max_reduction_kernel[grid](x, output)
        return output
