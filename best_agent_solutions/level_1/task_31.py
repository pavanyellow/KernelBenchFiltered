# level 1 index 31 agent name: KernelAgent o1 speedup: 1.04x

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class Model(nn.Module):
    """
    Simple model that performs an ELU activation (optimized with Triton).
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha

    @staticmethod
    @triton.jit
    def _elu_kernel(
        in_ptr,
        out_ptr,
        alpha,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        # Each program handles BLOCK_SIZE elements
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Load input (no mask needed since n_elements is a multiple of BLOCK_SIZE)
        x = tl.load(in_ptr + offsets)

        # ELU: out = x if x >= 0, else alpha*(exp(x) - 1)
        # Convert x to float32 for the exp operation in Triton
        x_f32 = x.to(tl.float32)
        out = tl.where(x_f32 >= 0.0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
        # Cast back to the same type as input
        out = out.to(x.dtype)

        # Store result (no mask needed for the same reason)
        tl.store(out_ptr + offsets, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor, using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        out = torch.empty_like(x)
        numel = x.numel()

        BLOCK_SIZE = 1024
        # Use a grid that covers the exact number of elements
        grid = (numel // BLOCK_SIZE,)

        self._elu_kernel[grid](
            x,
            out,
            self.alpha,
            numel,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out

# Helper functions to generate inputs matching the original specification
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    # Provide alpha value for initialization
    return [1.0]
