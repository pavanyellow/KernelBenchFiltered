# level 1 index 63 agent name: KernelAgent o1 speedup: 1.19x

import torch
import torch.nn as nn

# Enable faster convolution algorithms (may trade off determinism).
# In many cases, this speeds up 3×3 convolutions significantly.
torch.backends.cudnn.benchmark = True

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        # Keep default Conv2d init so weights match original code within floating tolerance
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to channels_last for potential cudnn speedups
        x = x.contiguous(memory_format=torch.channels_last)
        out = self.conv2d(x)
        return out
