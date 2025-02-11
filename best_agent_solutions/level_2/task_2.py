# level 2 index 2 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.61x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bias_clamp_scale_kernel(
    output_ptr, bias_ptr, scaling_factor,
    output_numel, bias_channels,
    height, width,
    BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel
    
    # Load and transform indices into n,c,h,w
    n = offsets // (bias_channels * height * width)
    tmp = offsets % (bias_channels * height * width)
    c = tmp // (height * width)
    tmp = tmp % (height * width)
    h = tmp // width
    w = tmp % width
    
    # Load values
    x = tl.load(output_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + c, mask=mask)
    
    # Fused operations
    x = x + b
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)  # clamp to [0,1]
    x = x * scaling_factor
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)  # clamp to [0,1] 
    x = x / scaling_factor
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        
    def forward(self, x):
        # Perform transposed convolution
        x = self.conv_transpose(x)
        
        # Get output dimensions
        batch_size, channels, height, width = x.shape
        numel = batch_size * channels * height * width
        
        # Launch triton kernel for fused operations
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
        fused_bias_clamp_scale_kernel[grid](
            x, self.bias, self.scaling_factor,
            numel, channels, height, width,
            BLOCK_SIZE=1024
        )
        
        return x
