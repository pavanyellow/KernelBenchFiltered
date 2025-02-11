# level 3 index 30 agent name: KernelAgent O3 Mini High speedup: 1.16x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Helper: _ntuple and to_2tuple
# ---------------------------------------------------------------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# ---------------------------------------------------------------------
# Custom CUDA kernels for window_partition, window_reverse, and patch_merging.
# We fuse the index arithmetic with memory load/store accesses as discussed.
# ---------------------------------------------------------------------
swin_ops_cpp_source = r"""
torch::Tensor window_partition_cuda(torch::Tensor x, int window_size);
torch::Tensor window_reverse_cuda(torch::Tensor windows, int window_size, int H, int W);
torch::Tensor patch_merging_cuda(torch::Tensor x, int H, int W, int C);
"""

swin_ops_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Kernel for window_partition:
// Input: x of shape [B, H, W, C]
// Each thread computes one element of the output windows tensor.
__global__ void window_partition_kernel(const float* __restrict__ x, float* __restrict__ out,
                                          int B, int H, int W, int C, int window_size) {
    int total = B * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Compute original indices.
    int c = idx % C;
    int tmp = idx / C;
    int w_idx = tmp % W;
    tmp /= W;
    int h_idx = tmp % H;
    int b = tmp / H;
    
    int num_tiles_h = H / window_size;
    int num_tiles_w = W / window_size;
    int tile_i = h_idx / window_size;
    int tile_j = w_idx / window_size;
    int local_i = h_idx % window_size;
    int local_j = w_idx % window_size;
    
    int window_idx = b * (num_tiles_h * num_tiles_w) + tile_i * num_tiles_w + tile_j;
    int out_idx = (((window_idx * window_size) + local_i) * window_size + local_j) * C + c;
    out[out_idx] = x[idx];
}

// ---------------------------------------------------------------------
// Kernel for window_reverse:
// Input: windows tensor of shape [num_windows, window_size, window_size, C]
// Each thread writes one element of the output tensor x (shape: [B, H, W, C]).
__global__ void window_reverse_kernel(const float* __restrict__ windows, float* __restrict__ out,
                                        int B, int H, int W, int C, int window_size) {
    int total = B * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int c = idx % C;
    int tmp = idx / C;
    int w_idx = tmp % W;
    tmp /= W;
    int h_idx = tmp % H;
    int b = tmp / H;
    
    int num_tiles_h = H / window_size;
    int num_tiles_w = W / window_size;
    int tile_i = h_idx / window_size;
    int tile_j = w_idx / window_size;
    int local_i = h_idx % window_size;
    int local_j = w_idx % window_size;
    
    int window_idx = b * (num_tiles_h * num_tiles_w) + tile_i * num_tiles_w + tile_j;
    int in_idx = ((window_idx * window_size + local_i) * window_size + local_j) * C + c;
    out[idx] = windows[in_idx];
}

// ---------------------------------------------------------------------
// Kernel for patch_merging:
// Input: x with shape [B, H, W, C]
// Output: out with shape [B, H/2, W/2, 4*C]
// For each output position, gather 4 pixels from x.
__global__ void patch_merging_kernel(const float* __restrict__ x, float* __restrict__ out,
                                       int B, int H, int W, int C) {
    int newH = H / 2;
    int newW = W / 2;
    int total = B * newH * newW * (4 * C);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int per_sample = newH * newW * (4 * C);
    int b = idx / per_sample;
    int r = idx % per_sample;
    int i = r / (newW * 4 * C);
    int r2 = r % (newW * 4 * C);
    int j = r2 / (4 * C);
    int k = r2 % (4 * C);
    int s = k / C;  // which patch among the 4 patches
    int c = k % C;
    // Determine offsets: for s==0 -> (0,0), s==1 -> (1,0), s==2 -> (0,1), s==3 -> (1,1)
    int row_offset = (s & 1);
    int col_offset = (s >> 1);
    int row = 2 * i + row_offset;
    int col = 2 * j + col_offset;
    int input_idx = (((b * H + row) * W) + col) * C + c;
    out[idx] = x[input_idx];
}

torch::Tensor window_partition_cuda(torch::Tensor x, int window_size) {
    // x: [B, H, W, C]
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);
    int num_tiles_h = H / window_size;
    int num_tiles_w = W / window_size;
    int num_windows = B * num_tiles_h * num_tiles_w;
    auto out = torch::empty({num_windows, window_size, window_size, C}, x.options());
    int total = B * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    window_partition_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, H, W, C, window_size);
    return out;
}

torch::Tensor window_reverse_cuda(torch::Tensor windows, int window_size, int H, int W) {
    // windows: [num_windows, window_size, window_size, C]
    auto num_windows = windows.size(0);
    int B = num_windows / ((H / window_size) * (W / window_size));
    int C = windows.size(3);
    auto out = torch::empty({B, H, W, C}, windows.options());
    int total = B * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    window_reverse_kernel<<<grid, block>>>(windows.data_ptr<float>(), out.data_ptr<float>(), B, H, W, C, window_size);
    return out;
}

torch::Tensor patch_merging_cuda(torch::Tensor x, int H, int W, int C) {
    // x: [B, H, W, C] -> out: [B, H/2, W/2, 4*C]
    int newH = H / 2;
    int newW = W / 2;
    auto B = x.size(0);
    auto out = torch::empty({B, newH, newW, 4 * C}, x.options());
    int total = B * newH * newW * (4 * C);
    int block = 256;
    int grid = (total + block - 1) / block;
    patch_merging_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, H, W, C);
    return out;
}
"""

swin_ops = load_inline(
    name="swin_ops",
    cpp_sources=swin_ops_cpp_source,
    cuda_sources=swin_ops_cuda_source,
    functions=["window_partition_cuda", "window_reverse_cuda", "patch_merging_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

# ---------------------------------------------------------------------
# Python wrappers for window_partition, window_reverse, and patch merging.
# If the input tensor is on CUDA, we call our custom CUDA kernels.
# ---------------------------------------------------------------------
def window_partition(x, window_size):
    # x: [B, H, W, C]
    if x.is_cuda:
        return swin_ops.window_partition_cuda(x.contiguous(), window_size)
    else:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

def window_reverse(windows, window_size, H, W):
    # windows: [num_windows, window_size, window_size, C]
    if windows.is_cuda:
        return swin_ops.window_reverse_cuda(windows, window_size, H, W)
    else:
        B = int(windows.shape[0] / ((H * W) / (window_size * window_size)))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

def patch_merging_op(x):
    # x: [B, H, W, C]
    if x.is_cuda:
        B, H, W, C = x.shape
        out = swin_ops.patch_merging_cuda(x.contiguous(), H, W, C)
        return out.view(B, -1, 4 * C)
    else:
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], -1)
        return x_cat.view(B, -1, 4 * C)

# ---------------------------------------------------------------------
# MLP Module: Two linear layers with activation and dropout.
# ---------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------------------------------------------------------------
# Window Attention Module.
#
# Computes self-attention within a window using PyTorch’s optimized linear layers,
# normalization, and softmax. The relative positional bias is computed via a small MLP.
# ---------------------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # tuple: (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Create relative coordinate table.
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        # Fuse sign and logarithm transformation as per our analysis.
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(8 * torch.abs(relative_coords_table) + 1.0) / 3
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Precompute relative position index.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize query and key vectors.
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        attn = (q_norm @ k_norm.transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.to(x.device), max=4.605170185988091).exp()
        attn = attn * logit_scale

        # Compute relative positional bias.
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---------------------------------------------------------------------
# Swin Transformer Block.
#
# Contains window-based self-attention (with optional cyclic shift)
# plus an MLP, along with residual connections and LayerNorm.
# ---------------------------------------------------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size)
        )
        self.drop_path = nn.Identity()  # For simplicity; replace with stochastic depth if desired.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Create attention mask for cyclic shift if needed.
        if self.shift_size > 0:
            H, W = self.input_resolution
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # Partition shifted feature map into windows.
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # Reverse windows back to feature map.
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

# ---------------------------------------------------------------------
# Patch Merging Module.
#
# Downsamples the feature map by fusing 4 neighboring patches.
# When on CUDA, we call our custom fused patch_merging kernel.
# ---------------------------------------------------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size for patch merging"
        x = x.view(B, H, W, C)
        if x.is_cuda:
            x = patch_merging_op(x)
        else:
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
            x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

# ---------------------------------------------------------------------
# BasicLayer: A stage consisting of several SwinTransformerBlocks and an optional downsampling.
# ---------------------------------------------------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# ---------------------------------------------------------------------
# PatchEmbed: Splits the input image into patches and projects them.
# ---------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution  # [H_patch, W_patch]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], "Input image size doesn't match"
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        if self.norm is not None:
            x = self.norm(x)
        return x

# ---------------------------------------------------------------------
# Main Model: Swin Transformer.
#
# Exposes the same Module interface as the original code.
# ---------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution  # [H_patch, W_patch]
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
