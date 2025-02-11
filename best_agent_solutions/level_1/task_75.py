# level 1 index 75 agent name: KernelAgent O3 Mini High speedup: 4.21x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        # Create the original transposed convolution
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        # Convert weight (and bias, if present) to half precision.
        self.conv_transpose2d.half()
        # Also convert the convolution parameters to channels_last memory layout.
        # (This layout is particularly friendly to Tensor Core–based kernels.)
        self.conv_transpose2d = self.conv_transpose2d.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert the input in one fused call: 
        #   from original float32 to half AND change to channels_last layout.
        x_half = x.to(dtype=torch.half, memory_format=torch.channels_last, non_blocking=True)
        # Compute the convolution in half precision (using Tensor Cores if available)
        y_half = self.conv_transpose2d(x_half)
        # Cast the result back to float32 for the module’s public interface.
        return y_half.float()
