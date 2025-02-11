# level 2 index 100 agent name: KernelAgent 4o speedup: 1.15x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define Triton kernel for clamping and dividing
@triton.jit
def clamp_and_divide_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    min_value,  # minimum value for clamping
    divisor,  # divisor for element-wise division
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr  # size of a block
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_clamped = tl.where(x < min_value, min_value, x)
    x_divided = x_clamped / divisor
    tl.store(out_ptr + offsets, x_divided, mask=mask)

def clamp_and_divide_torch(input, min_value, divisor):
    output = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    clamp_and_divide_kernel[grid](input, output, min_value, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = clamp_and_divide_torch(x, self.min_value, self.divisor)
        return x
