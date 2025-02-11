# level 3 index 8 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.73x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        self.has_fused_bn = False
        
        # Pre-compile forward paths
        self.compiled_eval_forward = None
        self.compiled_train_forward = None
        
        # Pre-allocate buffers for inference
        self.register_buffer('identity_buffer', None, persistent=False)

    def fuse_bn_into_conv(self):
        if self.has_fused_bn or self.training:
            return
            
        # Fuse first conv+bn
        w1 = self.conv1.weight
        mean1 = self.bn1.running_mean
        var1 = self.bn1.running_var
        gamma1 = self.bn1.weight
        beta1 = self.bn1.bias
        eps = self.bn1.eps
        
        w1 = w1 * (gamma1 / torch.sqrt(var1 + eps)).reshape(-1, 1, 1, 1)
        b1 = beta1 - mean1 * gamma1 / torch.sqrt(var1 + eps)
        
        conv1 = nn.Conv2d(self.conv1.in_channels, self.conv1.out_channels,
                         self.conv1.kernel_size, self.conv1.stride,
                         self.conv1.padding, bias=True)
        conv1.weight = nn.Parameter(w1)
        conv1.bias = nn.Parameter(b1)
        self.conv1 = conv1
        
        # Fuse second conv+bn
        w2 = self.conv2.weight
        mean2 = self.bn2.running_mean
        var2 = self.bn2.running_var
        gamma2 = self.bn2.weight
        beta2 = self.bn2.bias
        
        w2 = w2 * (gamma2 / torch.sqrt(var2 + eps)).reshape(-1, 1, 1, 1)
        b2 = beta2 - mean2 * gamma2 / torch.sqrt(var2 + eps)
        
        conv2 = nn.Conv2d(self.conv2.in_channels, self.conv2.out_channels,
                         self.conv2.kernel_size, self.conv2.stride,
                         self.conv2.padding, bias=True)
        conv2.weight = nn.Parameter(w2)
        conv2.bias = nn.Parameter(b2)
        self.conv2 = conv2
        
        # Fuse downsample conv+bn
        conv_down = self.downsample[0]
        bn_down = self.downsample[1]
        w_down = conv_down.weight
        mean_down = bn_down.running_mean
        var_down = bn_down.running_var
        gamma_down = bn_down.weight
        beta_down = bn_down.bias
        
        w_down = w_down * (gamma_down / torch.sqrt(var_down + eps)).reshape(-1, 1, 1, 1)
        b_down = beta_down - mean_down * gamma_down / torch.sqrt(var_down + eps)
        
        downsample = nn.Conv2d(conv_down.in_channels, conv_down.out_channels,
                              conv_down.kernel_size, conv_down.stride,
                              conv_down.padding, bias=True)
        downsample.weight = nn.Parameter(w_down)
        downsample.bias = nn.Parameter(b_down)
        self.downsample = downsample
        
        self.has_fused_bn = True

    def _eval_forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        
        return out
    
    def _train_forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

    def forward(self, x):
        if self.training:
            if self.compiled_train_forward is None:
                self.compiled_train_forward = torch.compile(self._train_forward)
            return self.compiled_train_forward(x)
        else:
            if not self.has_fused_bn:
                self.fuse_bn_into_conv()
                self.compiled_eval_forward = torch.compile(self._eval_forward)
            return self.compiled_eval_forward(x)
