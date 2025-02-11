# level 2 index 47 agent name: KernelAgent 4o speedup: 1.32x

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Enable cudnn benchmark to optimize convolution operations dynamically
torch.backends.cudnn.benchmark = True

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs, 'NUM_WARPS': nw, 'MULTI_BLOCKS_PER_PROG': mbp}, num_stages=3)
        for bs in [64, 128, 256, 512, 1024, 2048]  # Power of 2 block sizes
        for nw in [4, 8]  # Number of warps
        for mbp in range(1, 9)  # Number of blocks per program
    ],
    key=['N'],
)
@triton.jit
def mish_tanh_kernel(input_ptr, output_ptr, N, 
                     BLOCK_SIZE: tl.constexpr, 
                     NUM_WARPS: tl.constexpr,
                     MULTI_BLOCKS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * MULTI_BLOCKS_PER_PROG
    for i in range(MULTI_BLOCKS_PER_PROG):
        offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        softplus_x = tl.log(1 + tl.exp(x))
        mish_result = x * libdevice.tanh(softplus_x)
        result = libdevice.tanh(mish_result)
        tl.store(output_ptr + offsets, result, mask=mask)

def mish_tanh_triton(input_tensor):
    output_tensor = torch.empty_like(input_tensor)
    N = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE'] * meta['MULTI_BLOCKS_PER_PROG']),)
    mish_tanh_kernel[grid](input_tensor, output_tensor, N)
    return output_tensor

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = mish_tanh_triton(x)
        return x

def get_inputs():
    return [torch.randn(16, 3, 16, 32, 32)]

def get_init_inputs():
    return [3, 16, 3]
