# level 1 index 20 agent name: KernelAgent 4o speedup: 1.06x

import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def leaky_relu_kernel(X_ptr, Y_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for the block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boundary check
    mask = offsets < n_elements

    # Load input data from global memory
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Leaky ReLU
    y = tl.where(x >= 0, x, negative_slope * x)
    
    # Store the result back to global memory
    tl.store(Y_ptr + offsets, y, mask=mask)


class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        # Prepare output tensor
        output = torch.empty_like(x)
        # Grid configuration: Choose a suitable block size
        BLOCK_SIZE = 2048  # Adjusting block size for better performance
        n_elements = x.numel()
        grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        leaky_relu_kernel[grid](
            x,
            output,
            n_elements,
            self.negative_slope,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
