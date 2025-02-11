# level 2 index 40 agent name: KernelAgent O3 Mini High speedup: 2.70x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fixed dimensions:
#   M (batch size) = 128, N (out_features) = 128, K (in_features) = 64.
# Tile dimensions are chosen so that M, N, and K are evenly divided.
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32  # With K=64, BLOCK_K=32 makes two iterations.

# In this fused kernel we target the two adjacent operations which are easiest to fuse:
#   1. The linear (matrix multiplication and bias addition) operation.
#   2. The subsequent scaling (a residual multiplication with factor = 1+scaling_factor).
# This kernel therefore computes:
#    output = (x @ wᵀ + bias) * (1 + scaling_factor)
@triton.jit
def fused_linear_kernel(x_ptr, w_ptr, bias_ptr, out_ptr,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, factor: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles_n = N // BLOCK_N      # Number of tiles along the output (N) dimension.
    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n

    # Starting offsets for this tile.
    row_offset = tile_m * BLOCK_M
    col_offset = tile_n * BLOCK_N

    # Precompute index vectors for rows, columns, and the K-dimension tile.
    row_idx = tl.arange(0, BLOCK_M)           # (BLOCK_M,)
    col_idx = tl.arange(0, BLOCK_N)           # (BLOCK_N,)
    k_idx   = tl.arange(0, BLOCK_K)           # (BLOCK_K,)

    # For the input x (shape: [M, K]), each row starts at (row_index * K)
    x_row_off = (row_offset + row_idx) * K      # (BLOCK_M,)
    # For the weight matrix w (stored as [N, K]), each output channel has K elements.
    w_col_off = (col_offset + col_idx) * K        # (BLOCK_N,)

    # Initialize an accumulator in fp32 to preserve precision during dot products.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in chunks of BLOCK_K.
    for k in range(0, K, BLOCK_K):
        # Load a BLOCK_M x BLOCK_K tile from the input x.
        a = tl.load(x_ptr + x_row_off[:, None] + (k + k_idx)[None, :])
        # Load a BLOCK_K x BLOCK_N tile from the weight matrix.
        # Note: w is stored as (N, K), so we load a tile corresponding to output columns 
        # and then transpose the K dimension into the first index.
        w_tile = tl.load(w_ptr + w_col_off[None, :] + (k + k_idx)[:, None])
        # Accumulate the product.
        acc += tl.dot(a, w_tile)

    # Load the bias corresponding to these output columns.
    bias_tile = tl.load(bias_ptr + (col_offset + col_idx))
    # Fuse the bias addition and the scaling (i.e. residual addition) into one step.
    acc = (acc + bias_tile[None, :]) * factor

    # Store the computed tile into the output tensor.
    tl.store(out_ptr + ((row_offset + row_idx)[:, None] * N + (col_offset + col_idx)[None, :]), acc)


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        # nn.Linear stores weight with shape (out_features, in_features) and bias of shape (out_features,)
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Determine dimensions: M=batch size, K=in_features, N=out_features.
        M = int(x.shape[0])
        K = int(x.shape[1])
        N = int(self.matmul.out_features)

        # Instead of computing:
        #   y = linear(x)    [= x · wᵀ + bias]
        #   output = y + scaling_factor * y  (i.e. residual addition)
        # we fuse these two adjacent operations into one kernel.
        # The factor to multiply with is (1 + scaling_factor).
        factor = float(1.0 + self.scaling_factor)

        # Ensure inputs (x, weight, bias) are contiguous.
        x = x.contiguous()
        w = self.matmul.weight.contiguous()  # w has shape (N, K)
        b = self.matmul.bias.contiguous()      # bias has shape (N,)

        # Allocate output tensor.
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Each kernel instance computes a tile of size (BLOCK_M x BLOCK_N).
        grid = ((M // BLOCK_M) * (N // BLOCK_N),)
        fused_linear_kernel[grid](x, w, b, output, M, N, K, factor,
                                   BLOCK_M, BLOCK_N, BLOCK_K)
        return output


# Helper functions to provide input and initialization arguments.
batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
