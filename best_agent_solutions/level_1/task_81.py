# level 1 index 81 agent name: KernelAgent O3 Mini High speedup: 1.24x

import torch
import torch.nn as nn
import torch.jit

# Enable some CUDA configurations for optimal performance on NVIDIA GPUs.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

class ModelImpl(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel,
    supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelImpl, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

def Model(*args, **kwargs):
    """
    Factory function that returns a TorchScript scripted instance of ModelImpl.
    This ensures that when the module is constructed via:
    
        solution_model = Model(*inits)
    
    the returned model is a scripted nn.Module with identical behavior
    (within float tolerance) as the original implementation.
    """
    instance = ModelImpl(*args, **kwargs)
    return torch.jit.script(instance)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
