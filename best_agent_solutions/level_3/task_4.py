# level 3 index 4 agent name: KernelAgent 4o speedup: 2.20x

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5Optimized(nn.Module):
    def __init__(self, num_classes: int):
        super(LeNet5Optimized, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the already optimized PyTorch native operations in a scriptable manner
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def script_model(model):
    return torch.jit.script(model)

def Model(num_classes: int):
    model_instance = LeNet5Optimized(num_classes)
    return script_model(model_instance)

# Test code for the LeNet-5 optimized model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
