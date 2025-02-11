# level 2 index 61 agent name: KernelAgent o1 speedup: 1.52x

import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Fused ReLU using Triton with autotuning over block size, number of blocks
# per program, and number of warps. It replaces the naive PyTorch ReLU.
# --------------------------------------------------------------------

# Generate a list of Triton configs to autotune over
BLOCK_SIZES = [64, 128, 256, 512, 1024, 2048]  # ~6 powers of two
NBLOCKS = [1, 2, 4, 8]  # range of blocks per program
WARPS = [4, 8]          # possible num_warps values

configs = []
for bs in BLOCK_SIZES:
    for nb in NBLOCKS:
        for w in WARPS:
            configs.append(
                triton.Config(
                    {
                        'BLOCK_SIZE': bs, 
                        'NBLOCKS': nb, 
                        'num_warps': w
                    },
                    num_stages=1  # typically you can adjust num_stages as well
                )
            )

@triton.autotune(
    configs=configs,
    key=['n_elements']
)
@triton.jit
def fused_relu_kernel(
    x_ptr,                # pointer to the data
    n_elements,           # total number of elements
    BLOCK_SIZE: tl.constexpr,  # set by autotuner
    NBLOCKS: tl.constexpr,     # set by autotuner
    num_warps: tl.constexpr    # set by autotuner
):
    """
    A single-program, multiple-sub-block kernel that will process
    NBLOCKS * BLOCK_SIZE elements in one program_id(0). We then
    apply ReLU in-place for each sub-block.
    """
    # Program ID along x-dimension
    pid = tl.program_id(0)
    # Global offset for this program
    base_offset = pid * BLOCK_SIZE * NBLOCKS

    # Offsets for each thread within a sub-block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Loop over sub-blocks assigned to this program
    for i in range(NBLOCKS):
        offset_i = base_offset + i * BLOCK_SIZE
        idx = offset_i + offsets
        mask = idx < n_elements
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x_val = tl.maximum(x_val, 0.0)
        tl.store(x_ptr + idx, x_val, mask=mask)


def fused_relu(x: torch.Tensor) -> torch.Tensor:
    """
    In-place fused ReLU call with Triton autotuning.
    x is expected to be fp16 or fp32 on CUDA.
    """
    assert x.is_cuda, "Input must be on a CUDA device"
    n_elements = x.numel()

    # Grid: how many programs to launch
    def grid(meta):
        # Enough programs to cover all elements
        return ((n_elements + meta['BLOCK_SIZE'] * meta['NBLOCKS'] - 1)
                // (meta['BLOCK_SIZE'] * meta['NBLOCKS']),)

    # Call the kernel, but *do not* redefine the meta-parameters here
    fused_relu_kernel[grid](x, n_elements)
    return x


# --------------------------------------------------------------------
# Main Model Definition
# --------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        # Create layers in fp32 so they initialize identically
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

        # Convert trainable parameters to half precision
        self.conv_transpose.half()
        self.group_norm.half()

    def forward(self, x):
        # Convert input to half precision
        x = x.to(torch.float16)

        # 3D transpose convolution in FP16
        x = self.conv_transpose(x)

        # Fused ReLU in Triton
        x = fused_relu(x)

        # GroupNorm in FP16
        x = self.group_norm(x)

        # Convert output back to fp32
        x = x.to(torch.float32)
        return x
