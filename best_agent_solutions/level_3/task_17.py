# level 3 index 17 agent name: KernelAgent O3 Mini High speedup: 1.30x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        Creates a fused version of the model where the two convolution branches after
        the squeeze (one 1x1 and one 3x3) are merged into a single convolution.
        """
        super(Model, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Create the original expand conv layers (to get the same random initialization)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        
        # Now fuse the two expand branches into one convolution.
        # The fused expand convolution will have kernel size 3 (padding=1) and output channels equal to
        # expand1x1_channels + expand3x3_channels.
        self.expand_conv = nn.Conv2d(
            squeeze_channels,
            expand1x1_channels + expand3x3_channels,
            kernel_size=3,
            padding=1
        )
        
        self._fuse_expand_convs()
        
        # Once fused, the separate expand conv modules are no longer needed.
        del self.expand1x1, self.expand3x3

    def _fuse_expand_convs(self):
        with torch.no_grad():
            # === Fuse the expand1x1 branch ===
            # Get the weight and bias for the 1x1 conv branch
            weight_1x1 = self.expand1x1.weight  # shape: (C₁, squeeze_channels, 1, 1)
            bias_1x1 = self.expand1x1.bias      # shape: (C₁,)
            
            # Zero-pad weight_1x1 to shape (C₁, squeeze_channels, 3, 3)
            C1, Cin, _, _ = weight_1x1.shape
            padded_weight = weight_1x1.new_zeros((C1, Cin, 3, 3))
            # Place the original 1x1 weight in the center of a 3x3 kernel
            padded_weight[:, :, 1:2, 1:2] = weight_1x1
            
            # === Get the expand3x3 branch weight and bias ===
            weight_3x3 = self.expand3x3.weight  # shape: (C₃, squeeze_channels, 3, 3)
            bias_3x3 = self.expand3x3.bias      # shape: (C₃,)
            
            # === Fuse weights and biases ===
            fused_weight = torch.cat([padded_weight, weight_3x3], dim=0)
            self.expand_conv.weight.copy_(fused_weight)
            
            if self.expand_conv.bias is not None:
                fused_bias = torch.cat([bias_1x1, bias_3x3], dim=0)
                self.expand_conv.bias.copy_(fused_bias)

    def forward(self, x):
        # Apply squeeze conv followed by ReLU (making x nonnegative)
        x = self.relu(self.squeeze(x))
        # Apply the fused expand convolution and then a ReLU in-place.
        out = self.expand_conv(x)
        return F.relu(out, inplace=True)
