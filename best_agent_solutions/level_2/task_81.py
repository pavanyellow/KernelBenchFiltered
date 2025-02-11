# level 2 index 81 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.98x

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        # Block sizes increase in powers of 2, with varying blocks per program
        # Larger blocks = better throughput but more register pressure
        triton.Config({'BLOCK_SIZE': 64, 'BLOCKS_PER_PROGRAM': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCKS_PER_PROGRAM': 4}, num_warps=4), 
        triton.Config({'BLOCK_SIZE': 256, 'BLOCKS_PER_PROGRAM': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCKS_PER_PROGRAM': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCKS_PER_PROGRAM': 1}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_elementwise_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """
    Fused kernel for elementwise operations: swish/2.0 -> clamp -> tanh
    Args:
        x_ptr: Pointer to input tensor 
        out_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements per block (power of 2)
        BLOCKS_PER_PROGRAM: Number of blocks handled by each program
    """
    # Each program processes multiple blocks to reduce kernel launch overhead
    pid = tl.program_id(axis=0)
    block_start = pid * (BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    
    # Process multiple blocks sequentially within each program
    for block_idx in range(BLOCKS_PER_PROGRAM):
        # Calculate offsets and mask for this block
        offsets = block_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load input block
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Fused computation: swish/2.0 -> clamp -> tanh
        # 1. Compute swish and divide by 2 (using multiply for better perf)
        sigmoid_x = tl.sigmoid(x)
        swish_x = x * sigmoid_x
        divided = swish_x * 0.5  # faster than divide by 2.0
        
        # 2. Clamp to [-1, 1]
        clamped = tl.minimum(1.0, tl.maximum(-1.0, divided))
        
        # 3. Apply tanh using libdevice (optimized for [-1,1] range)
        result = libdevice.tanh(clamped)
        
        # Store result (no final clamp needed since tanh output is [-1,1])
        tl.store(out_ptr + offsets, result, mask=mask)

class Model(nn.Module):
    """
    Model that performs a linear transform followed by swish/2.0 -> clamp -> tanh.
    The elementwise operations are fused into a single kernel for maximum efficiency.
    
    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features
        bias (bool): Whether to include bias in linear layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Forward pass combines cuBLAS GEMM with fused elementwise operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # 1. Linear transform using cuBLAS
        x = self.gemm(x)
        
        # 2. Fused elementwise operations
        n_elements = x.numel()
        output = torch.empty_like(x)
        
        # Grid size calculation accounts for multiple blocks per program
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['BLOCKS_PER_PROGRAM']),)
        
        # Launch autotuned kernel for fused elementwise ops
        fused_elementwise_kernel[grid](
            x_ptr=x,
            out_ptr=output, 
            n_elements=n_elements,
        )
        
        return output
