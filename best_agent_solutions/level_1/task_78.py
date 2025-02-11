# level 1 index 78 agent name: KernelAgent o1 speedup: 1.27x

import torch
import torch.nn as nn

# For many use cases, enabling cudnn benchmarking can give faster kernels
torch.backends.cudnn.benchmark = True

class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias

        # Keep the same ConvTranspose2d module
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Convert weights to channels_last format for potential speedups
        self.conv_transpose2d.weight.data = self.conv_transpose2d.weight.data.contiguous(memory_format=torch.channels_last)
        if self.conv_transpose2d.bias is not None:
            self.conv_transpose2d.bias.data = self.conv_transpose2d.bias.data.contiguous()

        # We will JIT-trace a helper function after the first call
        self._forward_impl_traced = None

    def forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        # This method is what we trace with trace_module
        x = x.contiguous(memory_format=torch.channels_last)
        return self.conv_transpose2d(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On the first call, we create a traced version of forward_jit for the same shape and device.
        # This can significantly speed up subsequent calls due to optimized kernels.
        if self._forward_impl_traced is None:
            # Create a synthetic input of the same shape/dtype/device to do the tracing
            example_input = torch.zeros_like(x)
            # Trace the forward_jit method
            self._forward_impl_traced = torch.jit.trace_module(
                self,
                {"forward_jit": example_input}
            )

        # Call the traced version
        return self._forward_impl_traced.forward_jit(x)
