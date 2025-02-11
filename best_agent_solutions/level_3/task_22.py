# level 3 index 22 agent name: KernelAgent O3 Mini High speedup: 1.88x

import torch
import torch.nn as nn
import torch.nn.functional as F

# MBConv implements a Mobile Inverted Bottleneck Convolution.
# It always defines an expansion stage: when expand_ratio == 1 it simply acts as an identity.
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x += identity
        return x

# The unoptimized model _Model is defined as before.
class _Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Exported Model has the same interface as the original code.
#
# We try to optimize the module using torch.compile.
# However, because of a known inductor issue (raising a dataclass error during the first call),
# we force a warm‐up call with dummy input (backing up BN and parameter state before warmup and restoring afterward).
# If any error is raised during both the compile or warmup phase, we fall back to torch.jit.script.
#
# Also note that we use *args, **kwargs so that if get_init_inputs() is not empty, they will be forwarded to _Model.
def Model(*args, **kwargs):
    model = _Model(*args, **kwargs)
    try:
        # Attempt to compile the model with torch.compile.
        optimized_model = torch.compile(model, fullgraph=True)
        # Warm up the compiled model on a dummy input so that compilation is triggered.
        # We back up all state (parameters and buffers) so that a warm-up forward doesn’t change BN statistics.
        backup = {k: v.clone() for k, v in optimized_model.state_dict().items()}
        # Create a dummy input matching the expected shape. (We know the model will eventually see input of shape (N, 3, 224, 224); 
        # using batch size 1 is sufficient to trigger compilation.)
        dummy_input = torch.randn(1, 3, 224, 224, device=next(optimized_model.parameters()).device)
        optimized_model(dummy_input)
        # Restore the backed-up state (so that BN running stats and any buffers remain unaltered).
        optimized_model.load_state_dict(backup)
    except Exception:
        # Fall back to torch.jit.script if any error occurred during torch.compile or warm-up.
        optimized_model = torch.jit.script(model)
    return optimized_model
