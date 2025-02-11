# level 2 index 8 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.09x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Optimized version with simplified pooling and reduction operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        # Create conv layer but divide weights by divisor during init
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        with torch.no_grad():
            self.conv.weight.data = self.conv.weight.data / divisor
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data / divisor
        
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # Combined convolution and division (division folded into conv weights)
        x = self.conv(x)
        
        # Max pooling
        x = F.max_pool3d(x, self.pool_size)
        
        # Global average pooling over spatial dimensions (D,H,W)
        x = x.mean(dim=(-1,-2,-3), keepdim=True)
        
        # Add bias
        x = x + self.bias
        
        # Sum along specified dimension (typically channels)
        x = torch.sum(x, dim=self.sum_dim, keepdim=False)
        
        return x
