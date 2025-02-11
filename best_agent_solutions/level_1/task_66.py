# level 1 index 66 agent name: KernelAgent O3 Mini High speedup: 1.59x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel for converting FP32 (in contiguous NCDHW layout)
# into FP16 with channels_last_3d (NDHWC underlying layout).
#
# The idea is that the output tensor is allocated with torch.channels_last_3d,
# which means its underlying storage is ordered as if it has shape (N, D, H, W, C).
# Each thread converts one element from FP32 to FP16,
# performing the corresponding index remapping.
# -------------------------------------------------------------------------
@triton.jit
def fp32_to_fp16_channels_last_kernel(input_ptr, output_ptr,
                                        N: tl.constexpr, C: tl.constexpr,
                                        D: tl.constexpr, H: tl.constexpr,
                                        W: tl.constexpr, total_elements, 
                                        BLOCK_SIZE: tl.constexpr):
    # Each kernel invocation processes BLOCK_SIZE elements.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    idxs = pid * BLOCK_SIZE + offs  # Global indices in the (virtual) output space.
    mask = idxs < total_elements

    # The output’s underlying memory is in NDHWC order.
    # Given a linear index "idxs" in the range [0, N*D*H*W*C),
    # decode it as if the shape is (N, D, H, W, C):
    c = idxs % C
    tmp = idxs // C
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    tmp = tmp // H
    d = tmp % D
    n = tmp // D

    # Now compute the corresponding index into the input, which is in NCDHW order.
    # That is: offset = n * (C*D*H*W) + c * (D*H*W) + d * (H*W) + h * W + w.
    in_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w

    # Load FP32 values from the input tensor and cast them to FP16.
    x = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
    x_half = tl.cast(x, tl.float16)
    # Store the FP16 results into the output tensor.
    tl.store(output_ptr + idxs, x_half, mask=mask)


# -------------------------------------------------------------------------
# Triton kernel for converting a FP16 tensor elementwise to FP32.
#
# This kernel simply loads each FP16 value and casts it to FP32.
# -------------------------------------------------------------------------
@triton.jit
def fp16_to_fp32_kernel(input_ptr, output_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    idxs = pid * BLOCK_SIZE + offs
    mask = idxs < total_elements
    # Load FP16 values.
    x = tl.load(input_ptr + idxs, mask=mask)
    # Cast to FP32.
    x_fp32 = tl.cast(x, tl.float32)
    tl.store(output_ptr + idxs, x_fp32, mask=mask)


# -------------------------------------------------------------------------
# Host functions wrapping the Triton kernels.
# They mimic the interface of the previous inline CUDA conversion functions.
# -------------------------------------------------------------------------
def convert_fp32_to_fp16_channels_last(input: torch.Tensor) -> torch.Tensor:
    # Expect input to be in NCDHW contiguous layout, dtype=torch.float32.
    N, C, D, H, W = input.shape
    total = N * D * H * W * C  # total number of elements in the output (NDHWC order)
    # Allocate output as FP16 with channels_last_3d layout.
    output = torch.empty((N, C, D, H, W), device=input.device, dtype=torch.half,
                         memory_format=torch.channels_last_3d)
    BLOCK_SIZE = 1024  # Tunable block size

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fp32_to_fp16_channels_last_kernel[grid](input, output,
                                             N, C, D, H, W, total, 
                                             BLOCK_SIZE=BLOCK_SIZE)
    return output


def convert_fp16_to_fp32(input: torch.Tensor) -> torch.Tensor:
    # Input tensor is assumed to be FP16.
    total = input.numel()
    output = torch.empty_like(input, dtype=torch.float)
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fp16_to_fp32_kernel[grid](input, output, total, BLOCK_SIZE=BLOCK_SIZE)
    return output


# -------------------------------------------------------------------------
# Optimized Model definition using Triton conversion kernels.
# Maintains the same interface, initialization and behavior as the original.
# -------------------------------------------------------------------------
class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: tuple, 
        stride: tuple = (1, 1, 1), 
        padding: tuple = (0, 0, 0), 
        dilation: tuple = (1, 1, 1), 
        groups: int = 1, 
        bias: bool = False
    ):
        super(Model, self).__init__()
        # Create the standard 3D convolution layer.
        self.conv3d = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        # Convert Conv3d weights and (if applicable) bias to half precision,
        # and set their layout to channels_last_3d for optimized performance.
        self.conv3d = self.conv3d.to(dtype=torch.half, memory_format=torch.channels_last_3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Convert input from FP32 (NCDHW) to FP16 with channels_last_3d memory layout using the Triton kernel.
        x_converted = convert_fp32_to_fp16_channels_last(x)
        # Perform convolution in half precision.
        y = self.conv3d(x_converted)
        # Convert the result back to FP32.
        return convert_fp16_to_fp32(y)


# -------------------------------------------------------------------------
# Test configuration and helper functions.
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 256
width = 256

def get_inputs():
    # Create a GPU tensor with float32 dtype and standard NCDHW layout.
    x = torch.randn(batch_size, in_channels, depth, height, width, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    # Provide in_channels, out_channels, and kernel_size for initialization.
    return [in_channels, out_channels, kernel_size]

# -------------------------------------------------------------------------
# Quick test when executing as main.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate the model and move it to GPU.
    model = Model(in_channels, out_channels, kernel_size).cuda()
    # Retrieve test input.
    inputs = get_inputs()
    # Execute the forward pass.
    output = model(*inputs)
    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)
