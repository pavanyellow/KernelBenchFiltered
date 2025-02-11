# level 3 index 23 agent name: KernelAgent o1 speedup: 1.84x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Turn on cuDNN benchmark mode for faster conv kernels (especially if input shapes are fixed).
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True

###############################################################################
# 1) Inline CUDA code for fused batchnorm + ReLU (inference mode).
#    In training mode, we will fall back to the standard bn + relu.
#    This kernel handles an [N, C, H, W] input, normalizes it with
#    channel-wise (weight, bias, mean, var, eps), and optionally applies ReLU.
###############################################################################
_fused_bn_relu_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused BN + optional ReLU (inference mode only).
// x: (N*C*H*W) float
// out: (N*C*H*W) float
// weight, bias, mean, var: (C) float
// eps: scalar float
// relu_flag: bool (if true, apply ReLU after BN)
// N, C, H, W: dimensions
__global__ void fused_bn_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float eps,
    const bool relu_flag,
    const int N,
    const int C,
    const int H,
    const int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    if (idx < total_size) {
        // Which channel are we in?
        int c = (idx / (H * W)) % C;
        float inv_std = rsqrtf(running_var[c] + eps);
        float val = (x[idx] - running_mean[c]) * inv_std * weight[c] + bias[c];
        if (relu_flag && val < 0.f) {
            val = 0.f;
        }
        out[idx] = val;
    }
}

torch::Tensor fused_bn_relu_inference_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    bool relu_flag)
{
    // Expect x to be 4D: NCHW
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    int block_size = 256;
    int grid_size = (N * C * H * W + block_size - 1) / block_size;

    fused_bn_relu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        relu_flag,
        N, C, H, W
    );

    return out;
}
""".strip()

_fused_bn_relu_decl = r"""
torch::Tensor fused_bn_relu_inference_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    bool relu_flag);
""".strip()

# -----------------------------------------------------------------------------
# Compile the inline fused BN+ReLU module
# -----------------------------------------------------------------------------
_fused_bn_relu_module = load_inline(
    name="fused_bn_relu_inference",
    cpp_sources=_fused_bn_relu_decl,
    cuda_sources=_fused_bn_relu_src,
    functions=["fused_bn_relu_inference_cuda"],
    verbose=False,
)


###############################################################################
# 2) Helper function that calls into the fused CUDA kernel if not training.
#    If in training mode or on CPU, we revert to standard PyTorch BN + ReLU
#    for correctness and identical behavior with PyTorch's BatchNorm.
###############################################################################
def fused_bn_relu_inference(x, bn_module, relu_flag=True):
    """
    Fuses batchnorm + optional ReLU in inference mode via our custom CUDA kernel.
    If bn_module.training is True, or if x is not on CUDA, we fallback to
    bn + relu in standard PyTorch. 
    """
    if bn_module.training or (not x.is_cuda):
        # Fallback to vanilla BN + ReLU
        out = bn_module(x)
        return F.relu(out) if relu_flag else out
    else:
        # Inference mode -> use fused kernel
        return _fused_bn_relu_module.fused_bn_relu_inference_cuda(
            x,
            bn_module.weight,
            bn_module.bias,
            bn_module.running_mean,
            bn_module.running_var,
            bn_module.eps,
            relu_flag
        )


###############################################################################
# 3) A custom MBConv block that uses standard nn.Conv2d for conv layers
#    (thus leveraging cuDNN), but fuses BN + ReLU steps via the CUDA kernel
#    above when in eval mode. In training mode, it falls back to the usual
#    BN + ReLU in PyTorch for identical behavior.
###############################################################################
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        hidden_dim = round(in_channels * expand_ratio)

        # 1) Pointwise expand
        self.conv_expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(hidden_dim)

        # 2) Depthwise
        self.conv_depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                        groups=hidden_dim, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(hidden_dim)

        # 3) Pointwise project
        self.conv_project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # expand + BN + ReLU
        x = self.conv_expand(x)
        x = fused_bn_relu_inference(x, self.bn_expand, relu_flag=True)

        # depthwise + BN + ReLU
        x = self.conv_depthwise(x)
        x = fused_bn_relu_inference(x, self.bn_depthwise, relu_flag=True)

        # project + BN (no ReLU here in typical MobileNet/EfficientNet blocks)
        x = self.conv_project(x)
        x = fused_bn_relu_inference(x, self.bn_project, relu_flag=False)

        return x


###############################################################################
# 4) The overall Model, now using MBConvBlock and fused BN+ReLU calls
#    (for the first and last conv blocks) in inference mode. 
#    In training mode, it behaves the same as standard PyTorch BN + ReLU.
###############################################################################
class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Blocks
        self.mbconv1 = MBConvBlock(32, 16, 1, 1)
        self.mbconv2 = MBConvBlock(16, 24, 2, 6)
        self.mbconv3 = MBConvBlock(24, 40, 2, 6)
        self.mbconv4 = MBConvBlock(40, 80, 2, 6)
        self.mbconv5 = MBConvBlock(80, 112, 1, 6)
        self.mbconv6 = MBConvBlock(112, 192, 2, 6)
        self.mbconv7 = MBConvBlock(192, 320, 1, 6)

        # Head
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        Args:
            x: (tensor of shape [10, 3, 240, 240], float32) 
        Returns:
            (tensor of shape [10, num_classes], float32)
        """

        # Use channels_last memory format on CUDA for potential speedup:
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        # Stem conv + bn + relu
        x = self.conv1(x)
        x = fused_bn_relu_inference(x, self.bn1, relu_flag=True)

        # MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        # Head conv + bn + relu
        x = self.conv2(x)
        x = fused_bn_relu_inference(x, self.bn2, relu_flag=True)

        # Pool, flatten, linear
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
