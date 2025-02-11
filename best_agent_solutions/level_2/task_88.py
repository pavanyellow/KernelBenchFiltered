# level 2 index 88 agent name: KernelAgent O3 Mini High speedup: 2.38x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# This fused kernel has been re‐designed to process an entire sample (i.e. one GEMM row)
# at a time. Instead of launching one kernel instance per 64-element group (grid size = batch_size*num_groups),
# we launch one instance per sample (grid size = batch_size) and then fuse the per‐group computation via
# a reshape and reduction over a 2D array (shape: num_groups x group_size).
#
# Inside each kernel instance:
#  1. We load the full GEMM output row (1024 elements) with one vectorized load.
#  2. We reshape the 1D row into a 2D tensor of shape (num_groups, group_size) with num_groups=16 and group_size=64.
#  3. For each group (row of the 2D tensor) we compute mean and variance;
#     then we normalize the group and apply GroupNorm affine parameters.
#  4. We perform the first Swish activation:  swish1 = y * sigmoid(y).
#  5. We then multiply by the learned broadcast multiplier.
#  6. Finally, we perform the second Swish: output = (swish1 * multiplier) * sigmoid(swish1 * multiplier).
#
# All loads/stores are unmasked because 1024 (and 64) is a power‐of‑2 and our inputs are contiguous.
@triton.jit
def fused_groupnorm_swish_sample_kernel(
    y1_ptr,         # pointer to GEMM output (shape: [batch_size, n_features])
    out_ptr,        # pointer to output tensor (shape: [batch_size, n_features])
    gn_gamma_ptr,   # pointer to GroupNorm weight (gamma), shape: [n_features]
    gn_beta_ptr,    # pointer to GroupNorm bias (beta),  shape: [n_features]
    mweight_ptr,    # pointer to learned multiplier, shape: [n_features]
    eps: tl.constexpr,        # epsilon for numerical stability
    n_features: tl.constexpr, # total number of features (1024)
    num_groups: tl.constexpr, # number of groups (16)
    group_size: tl.constexpr  # features per group (64)
):
    # Each instance processes one sample.
    pid = tl.program_id(0)
    base = pid * n_features

    # Load the entire GEMM row (1024 contiguous elements).
    offs_all = tl.arange(0, n_features)
    row = tl.load(y1_ptr + base + offs_all)
    # Reshape the row to (num_groups, group_size) --> (16, 64)
    row = tl.reshape(row, (num_groups, group_size))

    # Compute per‐group mean and variance.
    # Sum over axis=1 (the group_size dimension).
    mean = tl.sum(row, axis=1) * (1.0 / group_size)  # shape: (num_groups,)
    sumsq = tl.sum(row * row, axis=1) * (1.0 / group_size)  # shape: (num_groups,)
    var = sumsq - mean * mean  # shape: (num_groups,)
    inv_std = 1.0 / tl.sqrt(var + eps)  # shape: (num_groups,)

    # Normalize each group; broadcast mean and inv_std (each of shape (num_groups,)) along axis1.
    norm = (row - mean[:, None]) * inv_std[:, None]

    # Load GroupNorm affine parameters (gamma and beta) and reshape to (num_groups, group_size)
    gamma = tl.load(gn_gamma_ptr + offs_all)
    gamma = tl.reshape(gamma, (num_groups, group_size))
    beta  = tl.load(gn_beta_ptr + offs_all)
    beta  = tl.reshape(beta, (num_groups, group_size))
    # Apply affine transformation.
    y = norm * gamma + beta

    # First Swish activation: y * sigmoid(y)
    swish1 = y * tl.sigmoid(y)

    # Load learned multiplier and reshape.
    m = tl.load(mweight_ptr + offs_all)
    m = tl.reshape(m, (num_groups, group_size))
    # Multiply by learned multiplier.
    z = swish1 * m

    # Second Swish activation: z * sigmoid(z)
    out_val = z * tl.sigmoid(z)

    # Flatten back to a 1D vector (length = n_features) and store the result.
    out_val = tl.reshape(out_val, (n_features,))
    tl.store(out_ptr + base + offs_all, out_val)


class Model(nn.Module):
    """
    Optimized Model that performs:
      1. A GEMM (Linear layer),
      2. Group Normalization (fused with affine parameters),
      3. A first Swish activation (x * sigmoid(x)),
      4. A broadcast multiplication by a learned parameter,
      5. A second Swish activation.
      
    Assumptions (for our single input shape optimization):
       - Input: tensor of shape (128, 512), dtype=torch.float32.
       - GEMM output: tensor of shape (128, 1024), dtype=torch.float32.
       - GroupNorm: features are split into num_groups groups
         (e.g. 16 groups so that each group has 1024 // 16 = 64 features).
       - Learned multiplier: shape (1024,).
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        # GEMM layer using PyTorch's Linear.
        self.gemm = nn.Linear(in_features, out_features)
        # GroupNorm with affine parameters.
        self.group_norm = nn.GroupNorm(num_groups, out_features, eps=1e-5, affine=True)
        # Learned multiplier parameter.
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        # Save constants for the fused kernel.
        self.out_features = out_features         # e.g., 1024
        self.num_groups = num_groups             # e.g., 16
        self.group_size = out_features // num_groups  # e.g., 64

    def forward(self, x):
        # x: shape (batch_size, in_features); e.g. (128, 512)
        # 1. Compute GEMM (linear layer).
        y1 = self.gemm(x).contiguous()  # shape: (batch_size, out_features)
        # 2. Allocate the output tensor.
        out = torch.empty_like(y1)
        batch_size = y1.shape[0]
        
        # 3. Launch the fused kernel with one program instance per sample.
        grid = (batch_size,)
        fused_groupnorm_swish_sample_kernel[grid](
            y1, out,
            self.group_norm.weight,  # GroupNorm gamma (shape: [1024])
            self.group_norm.bias,    # GroupNorm beta  (shape: [1024])
            self.multiply_weight,    # Learned multiplier (shape: [1024])
            float(self.group_norm.eps),  # Epsilon for numerical stability
            self.out_features,       # Total features, e.g., 1024
            self.num_groups,         # Number of groups, e.g., 16
            self.group_size          # Features per group, e.g., 64
        )
        return out


# Helpers to support the original interface.
batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
