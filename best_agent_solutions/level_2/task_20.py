# level 2 index 20 agent name: KernelAgent 4o speedup: 2.36x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def optimized_elementwise_triton(x_ptr, bias_ptr, out_ptr, ch_size, out_channels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (out_channels * tl.cdiv(ch_size, BLOCK_SIZE))
    channel_idx = (pid // tl.cdiv(ch_size, BLOCK_SIZE)) % out_channels
    block_idx = pid % tl.cdiv(ch_size, BLOCK_SIZE) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x_offset = batch_idx * out_channels * ch_size + channel_idx * ch_size + block_idx

    mask = block_idx < ch_size

    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    b = tl.load(bias_ptr + channel_idx)

    original = x
    result = ((x + b + original) * original) + original

    tl.store(out_ptr + x_offset, result, mask=mask)

def run_optimized_elementwise_triton(x, bias):
    batch_size, out_channels, depth, height, width = x.shape
    ch_size = depth * height * width
    out = torch.empty_like(x)

    BLOCK_SIZE = 512
    num_blocks = (ch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size * out_channels * num_blocks,)

    optimized_elementwise_triton[grid](x, bias, out, ch_size, out_channels, BLOCK_SIZE=BLOCK_SIZE)

    return out

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum,
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = run_optimized_elementwise_triton(x, self.bias)
        return x

# Constants for testing
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
