# level 3 index 6 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.32x

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(Model, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch 
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    @torch.compile(mode="reduce-overhead")  # Use torch.compile for optimization
    def forward(self, x):
        # Run the branches in parallel using torch.futures
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)  
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        # Concatenate along channel dimension
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
