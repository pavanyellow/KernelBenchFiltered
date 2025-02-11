# level 3 index 25 agent name: KernelAgent 4o speedup: 1.08x

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel to apply Swish activation function and elementwise addition
@triton.jit
def elementwise_relu_add_kernel(x_ptr, y_ptr, out_ptr, size,
                                BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Only store up to the required size
    mask = block_offsets < size

    # Load x, y
    x = tl.load(x_ptr + block_offsets, mask=mask)
    y = tl.load(y_ptr + block_offsets, mask=mask)

    # Compute ReLU(x) and ReLU(x) + y
    relu_x = tl.maximum(x, 0.0)
    out = relu_x + y
    tl.store(out_ptr + block_offsets, out, mask=mask)

def elementwise_relu_add(x, y):
    assert x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous()
    assert x.shape == y.shape
    out = torch.empty_like(x)
    size = x.numel()
    grid = (triton.cdiv(size, 1024),)
    elementwise_relu_add_kernel[grid](x, y, out, size, BLOCK_SIZE=1024)
    return out   

# Triton-based Channel Shuffle Kernel
@triton.jit
def channel_shuffle_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width, groups,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_channels_per_group = channels // groups

    # Calculate total elements per batch
    elements_per_batch = channels * height * width
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Linear index in the input tensor
    n = offset // elements_per_batch
    total_idx_within_batch = offset % elements_per_batch
    c = total_idx_within_batch // (height * width)
    hw_idx = total_idx_within_batch % (height * width)

    # Group indices
    group_id = c // num_channels_per_group
    inner_group_index = c % num_channels_per_group

    # Calculate the shuffled index
    shuffled_c = inner_group_index * groups + group_id
    output_idx = n * elements_per_batch + shuffled_c * height * width + hw_idx

    # Load from input and store to output
    tl.store(output_ptr + output_idx, tl.load(input_ptr + offset))

def channel_shuffle_triton(input, groups):
    batch_size, channels, height, width = input.shape
    output = torch.empty_like(input)
    grid = lambda META: (triton.cdiv(batch_size * channels * height * width, META['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        input, output, batch_size, channels, height, width, groups, BLOCK_SIZE=1024
    )
    return output

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        return channel_shuffle_triton(x, self.groups)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(Model, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffle(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shuffle(out)
        out = self.conv3(out)
        out = self.bn3(out)

        shortcut_out = self.shortcut(x)
        
        # Combine output with shortcut using the efficient Triton operation
        out = elementwise_relu_add(out, shortcut_out)
        return out
