# level 1 index 85 agent name: KernelAgent O3 Mini High speedup: 2.31x

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# New fused kernel that fuses the depthwise convolution and the bias addition.
@triton.jit
def fused_depthwise_conv_bias_kernel(x_ptr, w_ptr, bias_ptr, out_ptr,
                                     B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                                     out_H: tl.constexpr, out_W: tl.constexpr,
                                     kernel_h: tl.constexpr, kernel_w: tl.constexpr,
                                     total_kernel: tl.constexpr,
                                     BLOCK_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr):
    # Total number of output elements per (b, c) channel.
    out_size = out_H * out_W

    # Each kernel instance computes one (batch, channel) tile.
    # 'bc' indexes a unique (b, c) pair.
    bc = tl.program_id(0)
    # 'tile_idx' indexes a BLOCK_SIZE-wide tile within the flattened (b, c) output.
    tile_idx = tl.program_id(1)
    b = bc // C
    c = bc % C

    # Base pointers for the current (b, c) element.
    x_ptr_ch = x_ptr + b * (C * H * W) + c * (H * W)
    out_ptr_ch = out_ptr + b * (C * out_H * out_W) + c * (out_H * out_W)
    w_ptr_ch = w_ptr + c * (kernel_h * kernel_w)

    # Load bias if available.
    bias_val = tl.load(bias_ptr + c) if HAS_BIAS else 0.0

    # Compute the flattened output offsets for this tile.
    offs = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Use a mask for the store if the total number of outputs is not an exact multiple of BLOCK_SIZE.
    mask = offs < out_size

    # Map the flattened offsets into the 2D output coordinates.
    oh = offs // out_W
    ow = offs % out_W

    # Initialize the accumulator with the bias (fusing bias addition into initialization).
    acc = tl.full((BLOCK_SIZE,), bias_val, tl.float32)

    # Loop over each kernel element (flattened index).
    for k in tl.static_range(0, total_kernel):
        # Map the flattened kernel index into 2D coordinates.
        kh = k // kernel_w
        kw = k % kernel_w
        # Compute the corresponding input index.
        ind = (oh + kh) * W + (ow + kw)
        a = tl.load(x_ptr_ch + ind)
        w_val = tl.load(w_ptr_ch + k)
        acc += a * w_val

    # Write the computed tile back to global memory.
    tl.store(out_ptr_ch + offs, acc, mask=mask)

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, 
                 padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        # Only support depthwise convolution.
        # Thus, groups must equal in_channels and out_channels must equal in_channels.
        assert groups == in_channels, "Optimized Model only supports depthwise convolution (groups must equal in_channels)."
        assert out_channels == in_channels, "Optimized Model only supports depthwise convolution (out_channels must equal in_channels)."
        # Only support stride=1, padding=0, and dilation=1.
        assert stride_h == 1 and stride_w == 1, "Optimized Model supports only stride=1."
        assert padding_h == 0 and padding_w == 0, "Optimized Model supports only padding=0."
        assert dilation_h == 1 and dilation_w == 1, "Optimized Model supports only dilation=1."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w

        # Create the weight parameter with shape (in_channels, 1, kernel_size_h, kernel_size_w).
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias = None

        # Initialize parameters similarly to nn.Conv2d.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Assume the input is contiguous.
        B, C, H, W = x.shape
        kernel_h = self.kernel_size_h
        kernel_w = self.kernel_size_w

        # For stride=1, padding=0, dilation=1:
        #   out_H = H - kernel_h + 1 and out_W = W - kernel_w + 1.
        out_H = H - kernel_h + 1
        out_W = W - kernel_w + 1

        # Allocate the output tensor.
        out = torch.empty((B, C, out_H, out_W), device=x.device, dtype=x.dtype)

        # Choose a block size: the number of output elements per tile.
        BLOCK_SIZE = 256
        grid = (B * C, triton.cdiv(out_H * out_W, BLOCK_SIZE))

        # Squeeze the extra channel dimension of the weight, resulting in shape (C, kernel_h, kernel_w).
        weight = self.weight.squeeze(1).contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
            HAS_BIAS = 1
        else:
            # If bias is not used, create a dummy bias.
            bias = torch.empty((1,), device=x.device, dtype=x.dtype)
            HAS_BIAS = 0

        # Total number of kernel elements.
        total_kernel = kernel_h * kernel_w

        # Launch the new fused Triton kernel that fuses convolution and bias addition.
        fused_depthwise_conv_bias_kernel[grid](
            x, weight, bias, out,
            B, C, H, W,
            out_H, out_W,
            kernel_h, kernel_w,
            total_kernel,
            BLOCK_SIZE, HAS_BIAS
        )
        return out

# Test code (for validation)
batch_size = 16
in_channels = 3
out_channels = in_channels  # depthwise convolution
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    return [x]

def get_init_inputs():
    return [
        in_channels, 
        out_channels, 
        kernel_size_h, 
        kernel_size_w, 
        stride_h, 
        stride_w, 
        padding_h, 
        padding_w, 
        dilation_h, 
        dilation_w, 
        groups
    ]
