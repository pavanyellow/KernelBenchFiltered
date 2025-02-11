# level 1 index 28 agent name: KernelAgent o1 speedup: 1.02x

import torch
import torch.nn as nn
import triton
import triton.language as tl

#
# We need to match PyTorch's current definition of hardsigmoid, which is:
#   hardsigmoid(x) = clamp( (x + 3) / 6, 0, 1 )
# rather than clamp(0,1, 0.2*x + 0.5).
# That difference can lead to deviations up to ~0.0333, causing test failures.
#

@triton.jit
def _hardsigmoid_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask)
    # Compute hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)
    # Cast to float32 just in case, so that the arithmetic matches PyTorch more closely
    x_f32 = x.to(tl.float32)
    y = (x_f32 + 3.0) / 6.0
    y = tl.where(y > 1.0, 1.0, y)
    y = tl.where(y < 0.0, 0.0, y)
    # Store
    tl.store(out_ptr + offsets, y, mask=mask)

def hardsigmoid_triton(x: torch.Tensor) -> torch.Tensor:
    # Create an output tensor of the same size and type.
    out = torch.empty_like(x)
    # We'll treat the input as a 1D buffer of length n_elements.
    n_elements = x.numel()
    # Define a block size and grid size for triton.
    BLOCK_SIZE = 1024
    grid = lambda META: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch the kernel
    _hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE)
    return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton-accelerated hardsigmoid
        return hardsigmoid_triton(x)
