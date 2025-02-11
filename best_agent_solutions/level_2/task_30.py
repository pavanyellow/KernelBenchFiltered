# level 2 index 30 agent name: KernelAgent o1 speedup: 1.49x

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _groupnorm_hardtanh_fwd(
    x_ptr,            # (batch*C) float32
    y_ptr,            # (batch*C) float32
    weight_ptr,       # (C)       float32
    bias_ptr,         # (C)       float32
    eps,              # float
    act_min,          # float
    act_max,          # float
    BLOCK_SIZE: tl.constexpr,    # 512
    GROUPS: tl.constexpr,        # 8
    CHANNELS: tl.constexpr       # 512
):
    """
    Fused GroupNorm + HardTanh for [batch=128, channels=512].
    One block handles one batch element of 512 channels.
    """
    # current batch index
    batch_idx = tl.program_id(0)
    offset = batch_idx * CHANNELS
    
    # load input [512] float32
    c_idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset + c_idx)  # shape [512]

    # reshape x into (GROUPS, CHANNELS_PER_GROUP) => (8,64)
    x_2d = tl.reshape(x, (GROUPS, BLOCK_SIZE // GROUPS))  # (8,64)

    # compute mean
    mean = tl.sum(x_2d, axis=1) * (1.0 / (BLOCK_SIZE // GROUPS))  # shape [8]
    x_centered = x_2d - mean[:, None]

    # compute var -> inv_std
    var = tl.sum(x_centered * x_centered, axis=1) * (1.0 / (BLOCK_SIZE // GROUPS))
    inv_std = 1.0 / tl.sqrt(var + eps)  # shape [8]

    # load weight/bias
    wb_idx = tl.arange(0, BLOCK_SIZE)
    w = tl.load(weight_ptr + wb_idx)  # [512]
    b = tl.load(bias_ptr + wb_idx)    # [512]
    w_2d = tl.reshape(w, (GROUPS, BLOCK_SIZE // GROUPS))    # (8,64)
    b_2d = tl.reshape(b, (GROUPS, BLOCK_SIZE // GROUPS))    # (8,64)

    # combine inv_std and weight to reduce multiplications
    combined_factor = w_2d * inv_std[:, None]  # shape (8,64)

    # apply normalization, scale, shift
    x_out_2d = x_centered * combined_factor + b_2d  # shape (8,64)

    # hardtanh clamp
    x_out_2d = tl.minimum(tl.maximum(x_out_2d, act_min), act_max)

    # flatten
    x_out = tl.reshape(x_out_2d, (BLOCK_SIZE,))

    # store output
    tl.store(y_ptr + offset + c_idx, x_out)

class Model(nn.Module):
    """
    Optimized Model: Linear -> (Fused) GroupNorm + HardTanh using a custom Triton kernel.
    Keeps the same interface and random initialization as the original.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        # Linear (same initialization as before)
        self.gemm = nn.Linear(in_features, out_features)
        # GroupNorm
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        # HardTanh
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x: torch.Tensor):
        # 1. Linear
        x = self.gemm(x)  # shape [128, 512] in our fixed scenario

        # 2. Fused GroupNorm + HardTanh
        y = torch.empty_like(x, dtype=torch.float32)
        BLOCK_SIZE = x.shape[1]  # 512
        B = x.shape[0]           # 128
        grid = (B,)

        _groupnorm_hardtanh_fwd[grid](
            x, y,
            self.group_norm.weight, 
            self.group_norm.bias,
            self.group_norm.eps,
            self.hardtanh.min_val, 
            self.hardtanh.max_val,
            BLOCK_SIZE,                 # 512
            self.group_norm.num_groups, # 8 
            x.shape[1]                  # 512
        )

        return y
