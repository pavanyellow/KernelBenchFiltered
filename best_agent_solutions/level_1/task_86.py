# level 1 index 86 agent name: KernelAgent O3 Mini High speedup: 1.00x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        # Define the depthwise convolution with groups==in_channels.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        # Define the pointwise convolution.
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For training we use the unfused variant.
        if self.training:
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

        # In inference mode the two convolutions can be algebraically fused.
        # Let:
        #   w_depth: shape (C, 1, K, K)
        #   w_point: shape (out_channels, C, 1, 1)
        # We compute:
        #   combined_weight[oc, ic, i, j] = w_depth[ic, 0, i, j] * w_point[oc, ic, 0, 0]
        #
        # The effective bias is:
        #   If depthwise has bias (bias_dw) then for each oc:
        #       combined_bias[oc] = ∑_ic (w_point[oc, ic, 0, 0] * bias_dw[ic])
        #       Then if pointwise has its own bias (bias_pw) add it.
        #   Else, the bias is just pointwise.bias (which may be None).

        w_depth = self.depthwise.weight   # Shape: (C, 1, K, K)
        w_point = self.pointwise.weight     # Shape: (out_channels, C, 1, 1)
        # Compute combined weight.
        combined_weight = w_depth.unsqueeze(0) * w_point.unsqueeze(-1).unsqueeze(-1)
        combined_weight = combined_weight.view(
            self.pointwise.out_channels,
            self.depthwise.in_channels,
            self.depthwise.kernel_size[0],
            self.depthwise.kernel_size[1]
        )
        
        # Compute combined bias.
        use_bias_dw = self.depthwise.bias is not None
        use_bias_pw = self.pointwise.bias is not None
        if use_bias_dw:
            # w_point has shape (out_channels, C, 1, 1); squeeze the extra dimensions.
            combined_bias = (w_point.squeeze(-1).squeeze(-1) * self.depthwise.bias).sum(dim=1)
            if use_bias_pw:
                combined_bias = combined_bias + self.pointwise.bias
        else:
            combined_bias = self.pointwise.bias if use_bias_pw else None

        # Perform a single conv2d with the combined weight and bias.
        return F.conv2d(x, combined_weight, bias=combined_bias,
                        stride=self.depthwise.stride,
                        padding=self.depthwise.padding,
                        dilation=self.depthwise.dilation)
