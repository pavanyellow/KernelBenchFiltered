# level 2 index 70 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.51x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_scale_residual_kernel(
    output_ptr,  # Pointer to output buffer
    input_ptr,   # Pointer to input buffer (GEMM output)
    n_elements,  # Total number of elements
    scaling_factor, # Scalar multiplier
    BLOCK_SIZE: tl.constexpr,  # Size of parallel processing blocks
):
    # Calculate starting position for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset array for this block and corresponding mask
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor elements 
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute sigmoid(x)*scale + x all in one operation
    # Using tl.sigmoid for an optimized sigmoid implementation
    output = tl.sigmoid(x) * scaling_factor + x
    
    # Store result with mask
    tl.store(output_ptr + offsets, output, mask=mask)


class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Step 1: GEMM operation using PyTorch's optimized implementation
        gemm_out = self.gemm(x)
        
        # Step 2: Allocate output buffer for fused operations
        output = torch.empty_like(gemm_out)
        total_elements = gemm_out.numel()
        
        # Step 3: Configure grid and launch kernel
        grid = (triton.cdiv(total_elements, 128),)
        
        # Try different configurations based on input size
        if total_elements < 2048:
            block_size = 128
        elif total_elements < 8192:
            block_size = 256 
        else:
            block_size = 512
            
        fused_sigmoid_scale_residual_kernel[grid](
            output,                # Output buffer
            gemm_out,             # Input buffer
            total_elements,       # Total number of elements
            self.scaling_factor,  # Scaling factor
            BLOCK_SIZE=block_size,
        )
        
        return output
