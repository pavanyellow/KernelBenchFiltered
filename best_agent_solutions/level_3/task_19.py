# level 3 index 19 agent name: KernelAgent O3 Mini High speedup: 1.75x

import torch
import torch.nn as nn

# Helper functions to build convolution blocks.
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

# Internal implementation of the model.
class _Model(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(_Model, self).__init__()
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        # Flatten the tensor (batch_size, channels, 1, 1) -> (batch_size, channels)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Exported Model with the same interface as the original.
# To optimize the forward pass we try to use torch.compile.
# However, if torch.compile fails (as seen with a dataclass-related error),
# we fall back to torch.jit.script so that the module still runs correctly.
def Model(num_classes=1000, input_channels=3, alpha=1.0):
    model_instance = _Model(num_classes=num_classes, input_channels=input_channels, alpha=alpha)
    try:
        optimized_model = torch.compile(model_instance)
        return optimized_model
    except Exception:
        # Fallback: use TorchScript if torch.compile is not working correctly.
        return torch.jit.script(model_instance)
