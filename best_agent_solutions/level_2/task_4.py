# level 2 index 4 agent name: KernelAgent 4o speedup: 1.14x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_mish_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    
    # Define and apply Mish activation: mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)) = log(exp(x) + 1)

    # Implement softplus and tanh in terms of available ops
    ex = tl.exp(x)
    softplus_x = tl.log(ex + 1.0)
    tanh_softplus_x = (tl.exp(softplus_x) - tl.exp(-softplus_x)) / (tl.exp(softplus_x) + tl.exp(-softplus_x))
    mish_x = x * tanh_softplus_x

    # Apply Mish activation again
    ex_mish = tl.exp(mish_x)
    softplus_mish_x = tl.log(ex_mish + 1.0)
    tanh_softplus_mish_x = (tl.exp(softplus_mish_x) - tl.exp(-softplus_mish_x)) / (tl.exp(softplus_mish_x) + tl.exp(-softplus_mish_x))
    double_mish_x = mish_x * tanh_softplus_mish_x
    
    # Store result
    tl.store(y_ptr + offsets, double_mish_x, mask=mask)

def fused_mish_triton(x):
    N = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_mish_kernel[grid](x, y, N)  # Removed BLOCK_SIZE here to allow auto-tuning to manage it
    return y

class Model(nn.Module):
    """Simple model that performs a convolution, applies Mish, and another Mish."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = fused_mish_triton(x)
        return x

# Test settings
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
