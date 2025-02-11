# level 1 index 88 agent name: KernelAgent o1 speedup: 5.82x

import math
import torch
import torch.nn as nn

# Make sure Triton is installed:
#   pip install triton
# Or see https://github.com/openai/triton for details

import triton
import triton.language as tl
from triton.language.extra import libdevice  # We use libdevice.tanh instead of tl.math.tanh

# We'll implement a Triton kernel that fuses the GELU approximation into a single pass.
# The formula is:
#   GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * ( x + 0.044715*x^3 ) ))
#
# We'll define a Triton kernel that processes data in blocks of BLOCK_SIZE,
# loads each element, applies the formula once, then writes out the result.

@triton.jit
def _fused_gelu_kernel(
    x_ptr,           # points to input data (float32)
    out_ptr,         # points to output data (float32)
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    # Program ID is a one-dimensional grid.
    pid = tl.program_id(axis=0)
    # each block handles a slice of data of size BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    # compute the indices for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # mask to avoid out-of-bounds for loads and stores because 2000 is not a power of 2
    mask = offsets < n_elements

    # load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # approximate gelu: 
    #   gelu(x) = 0.5 * x * (1.0 + tanh( sqrt(2/pi)*( x + 0.044715*x^3 ) ))
    # 0.79788456 ~ sqrt(2/pi)
    x_c3 = x * x * x
    x_sum = x + 0.044715 * x_c3
    scaled = 0.79788456 * x_sum
    t = libdevice.tanh(scaled)  # use libdevice.tanh
    out_val = 0.5 * x * (1.0 + t)

    # store result
    tl.store(out_ptr + offsets, out_val, mask=mask)


def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the GELU approximation in a single fused Triton kernel on CUDA.
    On CPU or non-CUDA tensors, fall back to the original Python expression.
    Matches: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0)))).
    """
    if not x.is_cuda:
        # CPU fallback, same formula
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))

    # Allocate output tensor
    out = torch.empty_like(x)

    # number of elements total
    n_elements = x.numel()
    # define a block size
    BLOCK_SIZE = 1024

    # define grid
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    # launch kernel
    _fused_gelu_kernel[grid](
        x, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    return out


class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo 
    (identical to OpenAI GPT). 
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our fused function which performs all operations in one pass using Triton
        return fused_gelu(x)


batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
