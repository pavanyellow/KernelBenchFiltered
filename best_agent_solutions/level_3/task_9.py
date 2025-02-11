# level 3 index 9 agent name: KernelAgent o1 speedup: 1.10x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

###############################################################################
# 1) Fused CUDA Kernel: out = BatchNorm(x) + ReLU. 
#    (For simplicity, we implement only the inference (eval) version of BN, 
#    which uses running_mean, running_var, weight, bias, and does not update 
#    running stats or handle training-mode computations. In training mode, 
#    we will fall back on native PyTorch BN.)
#
#    If you want exactly matching behavior in training (including gradient 
#    flows and running stats updates), you'd need a more extensive custom BN 
#    kernel that incorporates those. Here, we demonstrate the fusion concept 
#    for the forward pass in inference.
###############################################################################
_fused_bn_relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// x, out: 4D tensors (N, C, H, W)
// weight, bias, running_mean, running_var: 1D (C)
// eps: float
// We assume x and out are float32 contiguous, same shape. 
// We do: 
//   val = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
//   val = max(0, val)     // ReLU
// for each element.

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps
){
    // total number of elements in x == N*C*H*W
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total){
        // figure out which channel 'c' this idx corresponds to
        // layout is (N, C, H, W), so channel can be extracted by:
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c = tmp % C;
        int n = tmp / C;

        float mean_c = running_mean[c];
        float var_c  = running_var[c];
        float w_c    = weight[c];
        float b_c    = bias[c];

        // offset to x[idx]
        float val = x[((n*C + c)*H + h)*W + w];

        // BN (inference mode)
        val = (val - mean_c) * rsqrtf(var_c + eps);
        val = val * w_c + b_c;

        // ReLU
        val = (val > 0.f) ? val : 0.f;

        out[((n*C + c)*H + h)*W + w] = val;
    }
}

torch::Tensor fused_bn_relu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
){
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    // shape of x: (N, C, H, W)
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    int total = N * C * H * W;
    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    fused_bn_relu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    return out;
}
""";

_fused_bn_relu_cpp_source = r"""
torch::Tensor fused_bn_relu_forward(
    torch::Tensor x, 
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
""";

_fused_bn_relu_module = load_inline(
    name="fused_bn_relu_module",
    cpp_sources=_fused_bn_relu_cpp_source,
    cuda_sources=_fused_bn_relu_source,
    functions=["fused_bn_relu_forward"],
    verbose=False
)

###############################################################################
# 2) Fused CUDA Kernel: out = ReLU(out + identity). (Already shown previously.)
###############################################################################
_fused_add_relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_relu_kernel(const float* __restrict__ x,
                                      const float* __restrict__ y,
                                      float* __restrict__ out,
                                      const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + y[idx];
        out[idx] = (val > 0.f) ? val : 0.f;
    }
}

torch::Tensor fused_add_relu_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same number of elements");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");

    auto out = torch::empty_like(x);
    const int size = x.numel();

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_add_relu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}
""";

_fused_add_relu_cpp_source = r"""
torch::Tensor fused_add_relu_forward(torch::Tensor x, torch::Tensor y);
""";

_fused_add_relu_module = load_inline(
    name="fused_add_relu_module",
    cpp_sources=_fused_add_relu_cpp_source,
    cuda_sources=_fused_add_relu_source,
    functions=["fused_add_relu_forward"],
    verbose=False
)

###############################################################################
# 3) Define the BasicBlock and Model classes, fusing certain elementwise ops.
#    - For BN+ReLU on the first conv in each BasicBlock, we use the custom 
#      kernel if we are in eval mode. If training is True, we fallback to 
#      official PyTorch BN + ReLU so that running stats and gradients remain correct.
#    - For the skip-connection add + ReLU, we call our fused_add_relu kernel.
###############################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        # If not training, fuse BN + ReLU into one kernel for speed. 
        if self.training:
            out = self.bn1(out)
            out = self.relu(out)
        else:
            # Inference path: fused BN+ReLU
            out = _fused_bn_relu_module.fused_bn_relu_forward(
                out, 
                self.bn1.weight, 
                self.bn1.bias, 
                self.bn1.running_mean, 
                self.bn1.running_var,
                self.bn1.eps
            )

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused add + ReLU:  out = ReLU(out + identity)
        out = _fused_add_relu_module.fused_add_relu_forward(out, identity)

        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the entire model.
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """

        # Top-level conv + BN + ReLU
        x = self.conv1(x)
        if self.training:
            x = self.bn1(x)
            x = self.relu(x)
        else:
            # Fuse BN+ReLU in eval mode for the top-level BN
            x = _fused_bn_relu_module.fused_bn_relu_forward(
                x, 
                self.bn1.weight, 
                self.bn1.bias, 
                self.bn1.running_mean, 
                self.bn1.running_var,
                self.bn1.eps
            )

        x = self.maxpool(x)

        # Stacks of residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pool
        x = self.avgpool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Final FC
        x = self.fc(x)

        return x

###############################################################################
# 4) Provide get_init_inputs() and get_inputs() to match original interface.
###############################################################################
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_init_inputs():
    """
    Returns the arguments used to initialize the Model class.
    """
    return [num_classes]

def get_inputs():
    """
    Returns the input tensor(s) for forward pass.
    """
    return [torch.randn(input_shape)]
