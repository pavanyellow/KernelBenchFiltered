# level 1 index 61 agent name: KernelAgent o1 speedup: 1.01x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Model, self).__init__()
        # There are no simple adjacent elementwise or normalization operations
        # to merge around the single ConvTranspose3d call. Hence, this version
        # remains functionally identical to the original.
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        # No adjacent operations to merge, so simply call ConvTranspose3d.
        return self.conv_transpose3d(x)
