# level 1 index 55 agent name: KernelAgent o1 speedup: 1.01x

import torch
import torch.nn as nn

# Enable a useful performance setting for convolutional networks
torch.backends.cudnn.benchmark = True

class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super(Model, self).__init__()
        # Same parameter initialization as the original:
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
        # Attempt to use torch.compile; if unavailable or fails, fallback
        try:
            self._compiled_forward = torch.compile(self._uncompiled_forward)
        except Exception:
            self._compiled_forward = self._uncompiled_forward

    def _uncompiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward logic
        return self.conv2d(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calls the compiled version (or fallback)
        return self._compiled_forward(x)
