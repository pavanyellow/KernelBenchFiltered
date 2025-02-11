# level 2 index 48 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.33x

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def fused_ops_kernel(
    input_ptr, scaling_factor_ptr, bias_ptr, output_ptr,
    total_elements, channels, depth, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes one block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Initialize offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices for the current position
    w = offsets % width
    h = (offsets // width) % height
    d = (offsets // (width * height)) % depth
    c = (offsets // (width * height * depth)) % channels
    
    # Load input and parameters
    x = tl.load(input_ptr + offsets, mask=mask)
    scale = tl.load(scaling_factor_ptr + c, mask=mask)
    b = tl.load(bias_ptr + c, mask=mask)
    
    # Apply operations
    x = x * scale
    x = libdevice.tanh(x)
    x = x * b
    x = tl.sigmoid(x)
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

class Model(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh,
    multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Get dimensions
        batch_size, channels, depth, height, width = x.shape
        total_elements = batch_size * channels * depth * height * width
        
        # Prepare grid and blocks
        BLOCK_SIZE = 1024  
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        # Allocate output
        output = torch.empty_like(x)
        
        # Launch kernel
        fused_ops_kernel[grid](
            x, self.scaling_factor, self.bias, output,
            total_elements, channels, depth, height, width,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
