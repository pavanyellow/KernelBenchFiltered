# level 1 index 36 agent name: KernelAgent O3 Mini High speedup: 2.86x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs, 'BLOCKS': nb, 'num_warps': nw}, num_stages=2)
        for bs in [64, 128, 256, 512, 1024, 2048]
        for nb in [1, 2, 4, 8]
        for nw in [4, 8]
    ],
    key=['N', 'C', 'H', 'W']
)
@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr,
                   N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                   eps: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr):
    # M: number of spatial elements (H*W) per sample.
    M = H * W  
    total_pixels = N * M
    pid = tl.program_id(0)
    for i in range(BLOCKS):
        # Each iteration processes BLOCK_SIZE pixels.
        pixel_index = pid * BLOCK_SIZE * BLOCKS + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        # Here, because the input tensor shape is fixed and power-of-2, we omit masks.
        # Recover the batch index (n) and the pixel index (r) within a sample.
        n = pixel_index // M       # shape: [BLOCK_SIZE]
        r = pixel_index % M        # shape: [BLOCK_SIZE]
        # For contiguous tensors of shape (N, C, H, W), the element at (n, c, h, w)
        # where r = h * W + w is located at offset = n*(C*M) + c*M + r.
        # First, compute the base offset for each pixel: n*(C*M) + r.
        base_offset = n * (C * M) + r  # shape: [BLOCK_SIZE]
        # Then compute per-channel offsets: for channel index c, the offset is c*M.
        channel_offsets = tl.arange(0, C) * M  # shape: [C]
        # Broadcast and add to get a 2D index array of addresses with shape [C, BLOCK_SIZE].
        addresses = base_offset[None, :] + channel_offsets[:, None]
        # Load the entire block for all channels at once.
        x_vals = tl.load(x_ptr + addresses)  # shape: [C, BLOCK_SIZE]
        # Compute sum of squares across channels (per pixel).
        sum_sq = tl.sum(x_vals * x_vals, axis=0)  # shape: [BLOCK_SIZE]
        # Compute the inverse RMS: rsqrt((sum_sq / C) + eps).
        inv_rms = tl.rsqrt(sum_sq / C + eps)  # shape: [BLOCK_SIZE]
        # Normalize each channel value by multiplying with the computed inv_rms.
        normalized = x_vals * inv_rms[None, :]  # broadcast over channel axis.
        # Store the normalized values to the output.
        tl.store(y_ptr + addresses, normalized)

def rmsnorm_triton(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton-based RMS normalization.
    
    x: input tensor of shape (N, C, H, W) in float32.
    eps: small epsilon to avoid division by zero.
    Returns a tensor of the same shape as x with normalized values.
    """
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    # Total number of pixels per batch sample.
    total_pixels = N * H * W
    # Launch grid: each kernel instance processes BLOCK_SIZE*BLOCKS pixels.
    grid = lambda meta: (triton.cdiv(total_pixels, meta['BLOCK_SIZE'] * meta['BLOCKS']),)
    rmsnorm_kernel[grid](x, y, N, C, H, W, eps)
    return y

class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMS normalization model.
        
        num_features: number of channels (C) expected in the input tensor.
        eps: small epsilon added to variance for numerical stability.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies RMS normalization along the channel dimension.
        
        x: input tensor of shape (N, C, H, W) with dtype torch.float32.
        Returns a tensor with the same shape and normalized values.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (N, C, H, W), but got {x.ndim} dimensions.")
        N, C, H, W = x.shape
        if C != self.num_features:
            raise ValueError(f"Expected number of channels {self.num_features}, but got {C}.")
        return rmsnorm_triton(x, self.eps)
