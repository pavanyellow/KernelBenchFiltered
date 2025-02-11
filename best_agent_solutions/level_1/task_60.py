# level 1 index 60 agent name: KernelAgent O3 Mini High speedup: 1.45x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton kernel: Fused FP32->FP16 conversion with layout reordering
# from standard [N, C, D, H, W] layout to channels_last_3d layout [N, D, H, W, C].
#
# We use the @triton.autotune decorator to search over candidate block sizes,
# number of blocks per program, and number of warps.
# ------------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Try around 5 power-of-2 candidate BLOCK_SIZE values with a couple of choices for N_BLOCKS and num_warps.
        triton.Config({'BLOCK_SIZE': 64,   'N_BLOCKS': 1, 'num_warps': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64,   'N_BLOCKS': 1, 'num_warps': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256,  'N_BLOCKS': 2, 'num_warps': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256,  'N_BLOCKS': 2, 'num_warps': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512,  'N_BLOCKS': 4, 'num_warps': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512,  'N_BLOCKS': 4, 'num_warps': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'N_BLOCKS': 6, 'num_warps': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'N_BLOCKS': 6, 'num_warps': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'N_BLOCKS': 8, 'num_warps': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, 'N_BLOCKS': 8, 'num_warps': 8}, num_warps=8),
    ],
    key=["total"]
)
@triton.jit
def fused_fp32_to_fp16_channels_last_3d_kernel(x_ptr, y_ptr, total, N, C, D, H, W, BLOCK_SIZE: tl.constexpr, N_BLOCKS: tl.constexpr):
    # Each program (i.e. kernel instance) processes BLOCK_SIZE*N_BLOCKS elements.
    pid = tl.program_id(0)
    # Compute how many programs (i.e. grid size) are needed.
    grid = tl.cdiv(total, BLOCK_SIZE * N_BLOCKS)
    # Loop over N_BLOCKS iterations per program to cover the entire tensor.
    for i in range(N_BLOCKS):
        # Compute the start offset for this iteration.
        curr_offset = (pid + i * grid) * BLOCK_SIZE
        # Generate indices for this BLOCK_SIZE chunk.
        offsets = curr_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        # Load a BLOCK_SIZE chunk from the FP32 input.
        x_vals = tl.load(x_ptr + offsets, mask=mask)
        # Convert FP32 -> FP16.
        y_vals = tl.cast(x_vals, tl.float16)

        # Decode linear index (offsets) into [N, C, D, H, W] coordinates.
        idx = offsets  # making a copy for coordinate extraction
        w = idx % W
        tmp = idx // W
        h = tmp % H
        tmp = tmp // H
        d = tmp % D
        tmp = tmp // D
        c_val = tmp % C
        n_val = tmp // C

        # Compute the output index for channels_last_3d layout [N, D, H, W, C]:
        # out_idx = n*(D*H*W*C) + d*(H*W*C) + h*(W*C) + w*C + c_val.
        out_idx = (((n_val * D + d) * H + h) * W + w) * C + c_val

        # Store the converted FP16 value into the output.
        tl.store(y_ptr + out_idx, y_vals, mask=mask)

def fused_fp32_to_fp16_channels_last_3d_launch(x: torch.Tensor) -> torch.Tensor:
    """
    Applies fused FP32 -> FP16 conversion with layout reordering.
    
    Args:
        x (torch.Tensor): Input FP32 tensor in standard [N, C, D, H, W] layout.
    
    Returns:
        torch.Tensor: Output FP16 tensor in channels_last_3d layout ([N, D, H, W, C]).
    """
    N, C, D, H, W = x.shape
    total = x.numel()
    # Allocate the output tensor in FP16 with channels_last_3d memory format.
    y = torch.empty(x.shape, dtype=torch.half, device=x.device, memory_format=torch.channels_last_3d)
    x_contig = x.contiguous()

    # Define the grid function using the autotuned meta-parameters.
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"] * meta["N_BLOCKS"]),)
    fused_fp32_to_fp16_channels_last_3d_kernel[grid](x_contig, y, total, N, C, D, H, W)
    return y

# ------------------------------------------------------------------------------
# Model: 3D Convolution with fused precision conversion and layout reordering.
#
# The forward pass has three stages:
# 1. Convert FP32 input (standard layout [N, C, D, H, W]) to FP16 in channels_last_3d format.
# 2. Run the FP16 convolution via cuDNN.
# 3. Convert the FP16 output back to FP32.
# ------------------------------------------------------------------------------
class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        # Create an FP32 convolution layer.
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
        # Immediately convert the convolution parameters to FP16.
        self.conv3d.weight.data = self.conv3d.weight.data.half()
        if bias:
            self.conv3d.bias.data = self.conv3d.bias.data.half()
        # Set the convolution layer to channels_last_3d memory format for optimal performance.
        self.conv3d = self.conv3d.to(memory_format=torch.channels_last_3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the convolution operation.

        Args:
            x (torch.Tensor): FP32 input tensor of shape [batch_size, in_channels, D, H, W].

        Returns:
            torch.Tensor: FP32 output tensor of shape [batch_size, out_channels, D_out, H_out, W_out].
        """
        # Stage 1: Convert FP32 input (standard layout) to FP16 in channels_last_3d format.
        x_fp16 = fused_fp32_to_fp16_channels_last_3d_launch(x)
        # Stage 2: Execute the convolution in FP16.
        out_fp16 = self.conv3d(x_fp16)
        # Stage 3: Convert the FP16 output back to FP32.
        out_fp32 = out_fp16.float()
        return out_fp32

# ------------------------------------------------------------------------------
# Test Code Interface (unchanged from the original)
# ------------------------------------------------------------------------------

# Fixed parameters for testing.
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel; note: input for 3D convolution is [B, C, D, H, W]
width = 64
height = 64
depth = 64

def get_inputs():
    """
    Prepares the input tensor.

    Returns:
         List[torch.Tensor]: A list with one tensor of shape [batch_size, in_channels, D, H, W] in FP32.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(batch_size, in_channels, depth, height, width, device=device)
    return [x]

def get_init_inputs():
    """
    Provides the initialization parameters for the Model.

    Returns:
         List: [in_channels, out_channels, kernel_size].
    """
    return [in_channels, out_channels, kernel_size]

# ------------------------------------------------------------------------------
# Optional test run when executing this script directly.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(*get_init_inputs()).to(device)
    inputs = [x.to(device) for x in get_inputs()]
    output = model(*inputs)
    print("Output shape:", output.shape)
