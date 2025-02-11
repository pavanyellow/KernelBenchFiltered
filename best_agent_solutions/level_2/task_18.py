# level 2 index 18 agent name: KernelAgent O3 Mini High speedup: 14.49x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# This fused kernel combines the linear operation and a full reduction over its outputs.
# In the original module the sequence of operations
#   x = self.linear(x)
#   x = torch.sum(x, dim=1, keepdim=True)
#   x = torch.max(x, dim=1, keepdim=True)[0]
#   x = torch.mean(x, dim=1, keepdim=True)
#   x = torch.logsumexp(x, dim=1, keepdim=True)
#   x = torch.logsumexp(x, dim=1, keepdim=True)
# mathematically collapses to computing:
#   output = sum_i (dot(x, weight[i, :]) + bias[i])
# Here we fuse the two adjacent operations that are easiest to merge:
# the linear layer and the subsequent summation reduction.
@triton.jit
def fused_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                 stride_x: tl.constexpr, stride_out: tl.constexpr,
                 in_features: tl.constexpr, out_features: tl.constexpr,
                 pad_in: tl.constexpr):
    # Each kernel instance processes one input row.
    pid = tl.program_id(0)
    
    # Compute pointers to the current row of input and output.
    x_row = x_ptr + pid * stride_x
    out_addr = out_ptr + pid * stride_out

    # Create a padded index range.
    offs = tl.arange(0, pad_in)
    # Load the input row with masking; no mask is used if in_features is already a power of 2.
    x_vals = tl.load(x_row + offs, mask=(offs < in_features), other=0.0)

    # Compute the aggregated weight vector by summing over all weight rows.
    # Each weight row corresponds to one linear output dimension.
    aggr_w = tl.zeros([pad_in], dtype=tl.float32)
    for i in range(out_features):
        w_vals = tl.load(weight_ptr + i * in_features + offs,
                         mask=(offs < in_features), other=0.0)
        aggr_w += w_vals

    # Dot product between the input row and the aggregated weight vector.
    dot_val = tl.sum(x_vals * aggr_w)

    # Sum all bias entries.
    aggr_bias = 0.0
    for i in range(out_features):
        aggr_bias += tl.load(bias_ptr + i)

    result = dot_val + aggr_bias
    tl.store(out_addr, result)


class Model(nn.Module):
    """
    Optimized Model that fuses an nn.Linear layer and subsequent reduction operations.
    
    Given the original operations:
         x = self.linear(x)
         x = torch.sum(x, dim=1, keepdim=True)
         x = torch.max(x, dim=1, keepdim=True)[0]
         x = torch.mean(x, dim=1, keepdim=True)
         x = torch.logsumexp(x, dim=1, keepdim=True)
         x = torch.logsumexp(x, dim=1, keepdim=True)
         
    Their net effect is equivalent to:
         output = sum_i (dot(x, weight[i, :]) + bias[i])
    This module exposes the same interface as before.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        # Use a standard linear layer for identical parameter initialization.
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        batch_size = x.shape[0]
        # Allocate the output tensor.
        output = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)

        # Ensure input and output tensors are contiguous.
        if not x.is_contiguous():
            x = x.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        # For a (batch_size, in_features) tensor, the row stride equals in_features.
        stride_x = x.stride(0)
        # For the output (batch_size, 1) tensor, the row stride is 1.
        stride_out = output.stride(0)

        # Compute the next power-of-2 padded size for in_features.
        pad_in = 1 << ((self.in_features - 1).bit_length())

        # Launch one kernel instance per input row.
        grid = (batch_size,)
        fused_kernel[grid](
            x,
            self.linear.weight,
            self.linear.bias,
            output,
            stride_x,
            stride_out,
            self.in_features,
            self.out_features,
            pad_in
        )
        return output


def get_inputs():
    # Returns a list containing a tensor on the GPU with shape (128, 10).
    return [torch.randn(128, 10, device='cuda')]


def get_init_inputs():
    # Returns the initialization parameters (in_features, out_features).
    return [10, 5]


if __name__ == '__main__':
    # Simple test run on CUDA.
    model = Model(10, 5).cuda()
    x = torch.randn(128, 10, device='cuda')
    y = model(x)
    print("Output shape:", y.shape)
