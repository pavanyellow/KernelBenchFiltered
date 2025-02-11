# level 2 index 53 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.45x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_hardtanh_gelu_kernel(
    output_ptr, input_ptr,
    n_elements,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # HardTanh
    x = tl.minimum(max_val, tl.maximum(min_val, x))

    # GELU Implementation using available operations
    # GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    # Using fastmath approximation: GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # We'll implement this using exp() since we don't have tanh directly
    # tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)
    
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    exp_2x = tl.exp(2 * inner)
    tanh_approx = (exp_2x - 1) / (exp_2x + 1)
    x = x * 0.5 * (1 + tanh_approx)

    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        # Apply scaling factor to weights and bias during initialization
        with torch.no_grad():
            self.gemm.weight.mul_(scaling_factor)
            if self.gemm.bias is not None:
                self.gemm.bias.mul_(scaling_factor)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.gemm(x)  # Scaling is built into the weights
        
        # Launch Triton kernel
        output = torch.empty_like(x)
        n_elements = x.numel()
        
        # Grid configuration
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Launch kernel
        fused_hardtanh_gelu_kernel[grid](
            output_ptr=output,
            input_ptr=x,
            n_elements=n_elements,
            min_val=self.hardtanh_min,
            max_val=self.hardtanh_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
