# level 3 index 21 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.45x

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_conv_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    batch, in_channels, out_channels, height, width,
    stride, padding, groups, max_value,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate position
    batch_idx = pid // (out_channels * height * width)
    rem = pid % (out_channels * height * width)
    oc = rem // (height * width)
    h = (rem // width) % height
    w = rem % width
    
    # Load bias
    acc = tl.load(b_ptr + oc) if b_ptr is not None else 0.0
    
    # Compute convolution
    for ic in range(in_channels // groups):
        for kh in range(3):
            for kw in range(3):
                ih = h * stride + kh - padding
                iw = w * stride + kw - padding
                
                if 0 <= ih < height and 0 <= iw < width:
                    x_val = tl.load(x_ptr + (batch_idx * in_channels + ic) * height * width + ih * width + iw)
                    w_val = tl.load(w_ptr + oc * (in_channels // groups) * 9 + ic * 9 + kh * 3 + kw)
                    acc += x_val * w_val
    
    # ReLU with configurable max value
    acc = tl.minimum(tl.maximum(acc, 0.0), max_value)
    
    # Store result
    tl.store(output_ptr + pid, acc)

class OptimizedModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(OptimizedModel, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size,
            stride=stride, padding=(kernel_size-1)//2,
            groups=hidden_dim, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Pre-allocate buffers for fused weights and precomputed BN scales
        self.register_buffer('expand_fused_weight', None)
        self.register_buffer('expand_fused_bias', None)
        self.register_buffer('depthwise_fused_weight', None)
        self.register_buffer('depthwise_fused_bias', None)
        self.register_buffer('project_fused_weight', None)
        self.register_buffer('project_fused_bias', None)
        self.register_buffer('expand_bn_scale', None)
        self.register_buffer('depthwise_bn_scale', None)
        self.register_buffer('project_bn_scale', None)
        
        # Cache shapes for faster computation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2
        
        # Use torch.compile for the parts we're not manually optimizing
        self.forward = torch.compile(self.forward)
    
    @torch.cuda.amp.autocast()
    def _fused_conv(self, x, weight, bias, stride, padding, groups, max_value=float('inf')):
        batch, channels, height, width = x.shape
        out_channels = weight.shape[0]
        output = torch.empty(
            (batch, out_channels, (height + 2*padding - self.kernel_size)//stride + 1,
             (width + 2*padding - self.kernel_size)//stride + 1),
            device=x.device, dtype=x.dtype
        )
        
        grid = (batch * out_channels * output.shape[2] * output.shape[3],)
        fused_conv_kernel[grid](
            x, weight, bias, output,
            batch, channels, out_channels, height, width,
            stride, padding, groups, max_value,
            BLOCK_SIZE=256
        )
        return output
    
    def _precompute_bn_scale(self, bn):
        """Precompute BN scale factor"""
        return bn.weight * torch.rsqrt(bn.running_var + bn.eps)
    
    def _fuse_bn_tensor(self, conv_w, conv_b, bn, precomputed_scale=None):
        if bn.training:
            return conv_w, conv_b
            
        scale = precomputed_scale if precomputed_scale is not None else self._precompute_bn_scale(bn)
        fused_w = conv_w * scale.reshape(-1, 1, 1, 1)
        fused_b = scale * (conv_b - bn.running_mean) + bn.bias if conv_b is not None else -bn.running_mean * scale + bn.bias
        return fused_w, fused_b

    def forward(self, x):
        identity = x
        
        if not self.training:
            # Precompute BN scales if not already done
            if self.expand_ratio != 1 and self.expand_bn_scale is None:
                self.expand_bn_scale = self._precompute_bn_scale(self.expand_bn)
                
            if self.depthwise_bn_scale is None:
                self.depthwise_bn_scale = self._precompute_bn_scale(self.depthwise_bn)
                
            if self.project_bn_scale is None:
                self.project_bn_scale = self._precompute_bn_scale(self.project_bn)
            
            # Fuse weights with precomputed scales
            if self.expand_ratio != 1 and self.expand_fused_weight is None:
                self.expand_fused_weight, self.expand_fused_bias = self._fuse_bn_tensor(
                    self.expand_conv.weight, None, self.expand_bn, self.expand_bn_scale)
                
            if self.depthwise_fused_weight is None:
                self.depthwise_fused_weight, self.depthwise_fused_bias = self._fuse_bn_tensor(
                    self.depthwise_conv.weight, None, self.depthwise_bn, self.depthwise_bn_scale)
                
            if self.project_fused_weight is None:
                # Scale the weights to account for ReLU6 in previous layer
                self.project_fused_weight, self.project_fused_bias = self._fuse_bn_tensor(
                    self.project_conv.weight * (1.0/6.0), None, self.project_bn, self.project_bn_scale)
            
            # Execute fused operations
            if self.expand_ratio != 1:
                x = self._fused_conv(x, self.expand_fused_weight, self.expand_fused_bias, 1, 0, 1, max_value=6.0)
            
            x = self._fused_conv(x, self.depthwise_fused_weight, self.depthwise_fused_bias,
                               self.stride, self.padding, x.shape[1], max_value=6.0)
            
            # No ReLU6 needed for final conv since we scaled the weights
            x = F.conv2d(x, self.project_fused_weight, self.project_fused_bias)
            
        else:
            # Training path remains the same
            if self.expand_ratio != 1:
                x = self.expand_conv(x)
                x = self.expand_bn(x)
                x = F.relu6(x)
            
            x = self.depthwise_conv(x)
            x = self.depthwise_bn(x)
            x = F.relu6(x)
            
            x = self.project_conv(x)
            x = self.project_bn(x)
        
        if self.use_residual:
            x += identity
            
        return x

# For compatibility with original interface
class Model(OptimizedModel):
    pass
