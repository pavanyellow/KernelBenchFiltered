# level 2 index 42 agent name: KernelAgent Claude 3.5 Sonnet speedup: 2.55x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCKS_PER_PROGRAM': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCKS_PER_PROGRAM': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCKS_PER_PROGRAM': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 32, 'BLOCKS_PER_PROGRAM': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCKS_PER_PROGRAM': 2}, num_stages=3, num_warps=8),
    ],
    key=['batch_size', 'out_channels', 'in_numel'],
)
@triton.jit
def fused_kernel(
    x_ptr, weight_ptr, output_ptr, bias_ptr,
    batch_size, in_channels, out_channels, 
    in_numel, flattened_kernel_size,
    stride_batch, stride_channel,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    # Program ID and block indices
    pid = tl.program_id(0)
    n_channel_blocks = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Each program handles BLOCKS_PER_PROGRAM consecutive channel blocks
    base_channel_block = (pid % n_channel_blocks) 
    batch_idx = pid // n_channel_blocks

    # Offsets for this block
    offs_m = base_channel_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Load bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < out_channels)

    # Load weight matrix block
    weight_block_ptr = weight_ptr + offs_m[:, None] * flattened_kernel_size + offs_k[None, :]
    
    # Iterate over input blocks
    for k in range(0, flattened_kernel_size, BLOCK_SIZE_K):
        k_remaining = min(BLOCK_SIZE_K, flattened_kernel_size - k)
        
        # Load weights
        w = tl.load(weight_block_ptr + k, 
                   mask=(offs_m[:, None] < out_channels) & (offs_k[None, :] < k_remaining))
        
        # Load and accumulate for multiple spatial locations
        x_batch_offset = batch_idx * stride_batch
        x = tl.load(x_ptr + x_batch_offset + k + offs_k, 
                   mask=offs_k < k_remaining)
        
        # Accumulate 
        acc += tl.sum(w * x[None, :], axis=1)

    # Normalize by spatial dimensions
    acc = acc / in_numel
    
    # Add bias
    acc = acc + bias
    
    # Store result
    out_offset = batch_idx * stride_channel
    tl.store(output_ptr + out_offset + offs_m, 
             acc, mask=offs_m < out_channels)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Save dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Reshape weight for faster access
        weight = self.conv_transpose.weight
        self.register_buffer('reshaped_weight', 
                           weight.reshape(out_channels, -1).contiguous())

    def forward(self, x):
        batch_size, _, height, width = x.shape
        out_height = height + self.kernel_size - 1
        out_width = width + self.kernel_size - 1
        in_numel = height * width
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)
        
        # Compute sizes
        flattened_kernel_size = self.kernel_size * self.kernel_size * self.in_channels
        
        # Prepare output
        intermediate = torch.empty((batch_size, self.out_channels), 
                                 device=x.device, dtype=x.dtype)
        
        # Launch kernel with autotuning
        grid = (batch_size * ((self.out_channels + 128 - 1) // 128),)  # Default size, will be adjusted by autotuner
        
        fused_kernel[grid](
            x_flat, self.reshaped_weight, intermediate, self.bias.squeeze(),
            batch_size, self.in_channels, self.out_channels,
            in_numel, flattened_kernel_size,
            x_flat.stride(0), intermediate.stride(0)
        )

        # Apply logsumexp and scaling
        return torch.logsumexp(intermediate, dim=1, keepdim=True) * 10.0

def get_inputs():
    batch_size = 128
    in_channels = 3 
    height, width = 32, 32
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    bias_shape = (out_channels, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]
