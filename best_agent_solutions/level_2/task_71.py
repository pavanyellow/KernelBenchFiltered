# level 2 index 71 agent name: KernelAgent 4o speedup: 1.18x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Optimized Triton kernel for element-wise multiplication and LeakyReLU
@triton.jit
def multiply_and_leaky_relu_kernel(
    input_ptr, output_ptr, reciprocal_divisor, negative_slope, numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Multiply and apply LeakyReLU
    val = input_data * reciprocal_divisor
    output_data = tl.where(val > 0.0, val, val * negative_slope)
    tl.store(output_ptr + offsets, output_data, mask=mask)

# Function to set up and call the Triton kernel
def multiply_and_leaky_relu_torch(input, reciprocal_divisor, negative_slope):
    output = torch.empty_like(input)
    numel = input.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    # Call the Triton kernel
    multiply_and_leaky_relu_kernel[grid](
        input_ptr=input, 
        output_ptr=output,
        reciprocal_divisor=reciprocal_divisor,
        negative_slope=negative_slope,
        numel=numel,
        BLOCK_SIZE=1024  # You can adjust this for different workloads
    )
    return output

class Model(nn.Module):
    """
    Optimized model using Triton for element-wise operations after a convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.reciprocal_divisor = 1.0 / divisor
        self.negative_slope = 0.01

    def forward(self, x):
        x = self.conv(x)
        x = multiply_and_leaky_relu_torch(x, self.reciprocal_divisor, self.negative_slope)
        return x

# Parameters for testing
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
