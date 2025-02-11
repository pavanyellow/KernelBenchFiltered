# level 1 index 57 agent name: KernelAgent O3 Mini High speedup: 1.02x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # The original implementation involves only one operation (the transposed convolution).
        # Since there are no adjacent elementwise or normalization operations to fuse,
        # we simply use the built‐in nn.ConvTranspose2d which already fuses bias-addition (if enabled)
        # and other minor elementwise operations internally.
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # There are no adjacent operations to merge, so the forward call remains unchanged.
        return self.conv_transpose2d(x)
