# level 1 index 54 agent name: O3 Mini Agent speedup: 1.00x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        # The only operation here is a conv3d. There are no adjacent elementwise or
        # normalization operations to merge, so we keep the conv3d call as-is.
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No adjacent operations exist to merge with conv3d.
        return self.conv3d(x)
