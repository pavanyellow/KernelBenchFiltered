# level 1 index 94 agent name: KernelAgent o1 speedup: 1.47x

import torch
import torch.nn as nn
import triton
import triton.language as tl

################################################################################
# A Triton kernel that accumulates sum of squared differences in one pass using
# atomic adds into a single partial sum tensor.
################################################################################
@triton.jit
def mse_kernel(
    predictions_ptr,    # pointer to float32 predictions
    targets_ptr,        # pointer to float32 targets
    partial_sum_ptr,    # pointer to float32 partial sum, length=1
    n_elements,         # number of elements to process
    BLOCK_SIZE: tl.constexpr
):
    # This program will process a contiguous block of BLOCK_SIZE elements
    # starting at an offset in [0..grid(0)].
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets under the valid mask
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute the squared differences and reduce them within the block
    diff = p - t
    sq = diff * diff
    block_sum = tl.sum(sq, axis=0)

    # Accumulate into a single partial sum atomically (partial_sum_ptr[0])
    tl.atomic_add(partial_sum_ptr, block_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    A function that uses a Triton kernel to compute the Mean Squared Error:
    mean((predictions - targets)^2).
    """
    assert predictions.is_cuda, "predictions must be a CUDA tensor."
    assert targets.is_cuda, "targets must be a CUDA tensor."
    assert predictions.numel() == targets.numel(), \
        "predictions and targets must have the same number of elements."
    assert predictions.dtype == torch.float32 and targets.dtype == torch.float32, \
        "This kernel only supports float32 tensors."

    size = predictions.numel()

    # Allocate a device tensor to collect the partial sum
    partial_sum = torch.zeros((1,), device=predictions.device, dtype=predictions.dtype)

    # Grid definition: number of blocks needed
    BLOCK_SIZE = 1024
    def grid(meta):
        return ( (size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

    # Launch our Triton kernel
    mse_kernel[grid](
        predictions,    # pointer to predictions data
        targets,        # pointer to targets data
        partial_sum,    # pointer to partial sum
        size,           # number of elements
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Divide by total size to get the mean, and return the scalar
    out = partial_sum / float(size)
    return out.reshape([])  # shape () scalar


################################################################################
# The optimized Model class
################################################################################

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks
    using a fused Triton kernel.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        # Calls our custom Triton MSE operator
        return triton_mse(predictions, targets)


##############################################################################
# For completeness, providing these helper functions as shown in the original
# code snippet. They are not strictly needed for the core logic but were part
# of the original interface.
##############################################################################

batch_size = 128
input_shape = (4096,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape, device='cuda'), 
            torch.randn(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    return []
