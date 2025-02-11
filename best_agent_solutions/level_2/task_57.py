# level 2 index 57 agent name: KernelAgent 4o speedup: 1.69x

import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return fused_relu_clamp(x)

@triton.jit
def relu_clamp_kernel(x_ptr, output_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load the input data
    x = tl.load(x_ptr + offsets)
    # Apply the fused ReLU, scale, clamp, and multiply operations
    relu_x = tl.where(x > 0, x, 0.0)  # Replace tl.max with tl.where to perform ReLU
    scaled_x = (relu_x + 3.0) / 6.0
    clamped_x = tl.clamp(scaled_x, 0.0, 1.0)
    result = relu_x * clamped_x
    # Store the result to output
    tl.store(output_ptr + offsets, result)

def fused_relu_clamp(x):
    # Calculate number of elements
    n_elements = x.numel()
    # Prepare an output tensor
    output = torch.empty_like(x)
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    relu_clamp_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
