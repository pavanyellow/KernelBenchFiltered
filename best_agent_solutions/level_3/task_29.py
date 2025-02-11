# level 3 index 29 agent name: KernelAgent O3 Mini High speedup: 2.08x

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import torch.utils.checkpoint as checkpoint
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------------------
# Fused CUDA kernel for Swin-MLP spatial operation --
# (Fuses window partitioning, a grouped 1×1 “spatial MLP” and window reversal)
# ------------------------------------------------------------------------------
cpp_source = r'''
#include <torch/extension.h>
torch::Tensor swin_mlp_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                               int ws, int num_heads, int pad_t, int pad_l);
'''

cuda_source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Preload constant offset arrays for a 7x7 window into constant memory.
__constant__ int off_h_7[49] = {
    0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,
    6,6,6,6,6,6,6
};
__constant__ int off_w_7[49] = {
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6
};

__global__ void swin_mlp_forward_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          int B, int H, int W, int C,
                                          int ws, int num_heads, int pad_t, int pad_l) {
    int total = B * H * W * C;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        // Map linear index to coordinate (b, h_out, w_out, c).
        int c = idx % C;
        int temp = idx / C;
        int w_out = temp % W;
        temp /= W;
        int h_out = temp % H;
        int b = temp / H;
    
        // Shifted coordinates with padding.
        int ph = h_out + pad_t;
        int pw = w_out + pad_l;
    
        int area = ws * ws;
        int win_row = ph / ws;
        int win_col = pw / ws;
        int base_h = win_row * ws;
        int base_w = win_col * ws;
        int local = (ph - base_h) * ws + (pw - base_w);
    
        int head_dim = C / num_heads;  // assume C divisible by num_heads.
        int group = c / head_dim;
    
        int group_area = group * area;
        int row_offset = (group_area + local) * area;
        int H_end = pad_t + H;
        int W_end = pad_l + W;
    
        float res = __ldg(&bias[group_area + local]);
    
        if (ws == 7) {
            // Unrolled loop for ws==7 using constant memory offsets.
            #pragma unroll
            for (int k = 0; k < 49; ++k) {
                int cur_h = base_h + off_h_7[k];
                int cur_w = base_w + off_w_7[k];
                float in_val = 0.f;
                if (cur_h >= pad_t && cur_h < H_end && cur_w >= pad_l && cur_w < W_end) {
                    int in_h = cur_h - pad_t;
                    int in_w = cur_w - pad_l;
                    int x_index = ((b * H + in_h) * W + in_w) * C + c;
                    in_val = __ldg(&x[x_index]);
                }
                int weight_index = row_offset + k;
                float w_val = __ldg(&weight[weight_index]);
                res += w_val * in_val;
            }
        } else {
            // Generic path for arbitrary window sizes.
            for (int k = 0; k < area; ++k) {
                int offset_h = k / ws;
                int offset_w = k % ws;
                int cur_h = base_h + offset_h;
                int cur_w = base_w + offset_w;
                float in_val = 0.f;
                if (cur_h >= pad_t && cur_h < H_end && cur_w >= pad_l && cur_w < W_end) {
                    int in_h = cur_h - pad_t;
                    int in_w = cur_w - pad_l;
                    int x_index = ((b * H + in_h) * W + in_w) * C + c;
                    in_val = __ldg(&x[x_index]);
                }
                int weight_index = row_offset + k;
                float w_val = __ldg(&weight[weight_index]);
                res += w_val * in_val;
            }
        }
    
        int out_index = ((b * H + h_out) * W + w_out) * C + c;
        y[out_index] = res;
    }
}

at::Tensor swin_mlp_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                            int ws, int num_heads, int pad_t, int pad_l) {
    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    int C = x.size(3);
    auto output = torch::empty_like(x);
    int total = B * H * W * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    swin_mlp_forward_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                                   output.data_ptr<float>(),
                                                   weight.data_ptr<float>(),
                                                   bias.data_ptr<float>(),
                                                   B, H, W, C, ws, num_heads, pad_t, pad_l);
    return output;
}
'''

swin_mlp_cuda = load_inline(
    name="swin_mlp_cuda",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["swin_mlp_forward"],
    verbose=False,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# Thin Python wrapper for the fused Swin-MLP spatial op.
def _swin_mlp_forward(x, weight, bias, ws, num_heads, pad_t, pad_l):
    if not x.is_cuda:
        raise RuntimeError("swin_mlp_forward: x must be a CUDA tensor")
    return swin_mlp_cuda.swin_mlp_forward(x, weight, bias, ws, num_heads, pad_t, pad_l)

# ------------------------------------------------------------------------------
# Fused CUDA kernel for Patch Merging --
# This kernel fuses the slicing and channel concatenation that occurs in PatchMerging.
# (Input: (B, H, W, C);  Output: (B, H//2, W//2, 4*C))
# ------------------------------------------------------------------------------
patch_merging_cpp_source = r'''
#include <torch/extension.h>
torch::Tensor patch_merging_forward(torch::Tensor x, int H, int W, int C);
'''

patch_merging_cuda_source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void patch_merging_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                               int B, int H, int W, int C) {
    // New spatial dimensions.
    int new_H = H / 2;
    int new_W = W / 2;
    // Total elements in the output tensor.
    int total = B * new_H * new_W * 4 * C;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        // Decompose idx into indices for b, h_out, w_out, and channel index (cc) in output.
        int tmp = idx;
        int cc = tmp % (4 * C);
        tmp /= (4 * C);
        int w_out = tmp % new_W;
        tmp /= new_W;
        int h_out = tmp % new_H;
        int b = tmp / new_H;
        // Determine which of the four parts and the corresponding channel.
        int part = cc / C;  // 0,1,2,3 correspond to x0,x1,x2,x3 respectively.
        int c = cc % C;
        // Calculate the corresponding input spatial indices.
        int in_h = 2 * h_out + ((part == 1 || part == 3) ? 1 : 0);
        int in_w = 2 * w_out + ((part == 2 || part == 3) ? 1 : 0);
        int x_idx = ((b * H + in_h) * W + in_w) * C + c;
        y[idx] = __ldg(&x[x_idx]);
    }
}

torch::Tensor patch_merging_forward(torch::Tensor x, int H, int W, int C) {
    int B = x.size(0);
    int new_H = H / 2;
    int new_W = W / 2;
    auto y = torch::empty({B, new_H, new_W, 4 * C}, x.options());
    int total = B * new_H * new_W * 4 * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    patch_merging_forward_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, H, W, C);
    return y;
}
'''

patch_merging_cuda = load_inline(
    name="patch_merging_cuda",
    cpp_sources=[patch_merging_cpp_source],
    cuda_sources=[patch_merging_cuda_source],
    functions=["patch_merging_forward"],
    verbose=False,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

def _patch_merging_forward(x, H, W, C):
    if not x.is_cuda:
        raise RuntimeError("patch_merging_forward: x must be a CUDA tensor")
    return patch_merging_cuda.patch_merging_forward(x, H, W, C)

# ------------------------------------------------------------------------------
# Model Modules
# ------------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size, window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
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
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"
        self.norm1 = norm_layer(dim)
        # Use a dummy Conv1d to hold the learned spatial MLP parameters.
        self.spatial_mlp = nn.Conv1d(self.num_heads * (self.window_size ** 2),
                                     self.num_heads * (self.window_size ** 2),
                                     kernel_size=1, groups=self.num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: (B, L, C) with L = H * W.
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # For shifted windows, simulate padding: pad top and left = window_size - shift_size.
        if self.shift_size > 0:
            pad_t = self.window_size - self.shift_size
            pad_l = self.window_size - self.shift_size
        else:
            pad_t = 0
            pad_l = 0
        # Call the fused CUDA kernel for spatial MLP.
        x = _swin_mlp_forward(x, self.spatial_mlp.weight, self.spatial_mlp.bias,
                               self.window_size, self.num_heads, pad_t, pad_l)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        # x: (B, L, C) with L = H * W.
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        if x.is_cuda:
            # Fuse the slicing and concatenation in one kernel.
            x_view = x.view(B, H, W, C)
            out = _patch_merging_forward(x_view, H, W, C)
            out = out.view(B, -1, 4 * C)
        else:
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            out = torch.cat([x0, x1, x2, x3], -1)
            out = out.view(B, -1, 4 * C)
        out = self.norm(out)
        out = self.reduction(out)
        return out

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) \
                          if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
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

# ------------------------------------------------------------------------------
# Helper functions for the evaluation harness.
# These functions expose the same interface as the original code.
# ------------------------------------------------------------------------------

def get_init_inputs():
    # Returns the tuple of arguments to initialize the Model.
    # (img_size, patch_size, in_chans, num_classes, embed_dim, depths,
    #  num_heads, window_size, mlp_ratio, drop_rate, drop_path_rate, norm_layer,
    #  patch_norm, use_checkpoint)
    return (224, 4, 3, 1000, 96, [2, 2, 6, 2], [3, 6, 12, 24], 7, 4.0, 0.0, 0.1, nn.LayerNorm, True, False)

def get_inputs():
    # Returns the input tensor tuple for model.forward.
    # The input is a float32 tensor of shape (10, 3, 224, 224).
    return (torch.randn(10, 3, 224, 224),)
