# level 1 index 17 agent name: KernelAgent 4o speedup: 1.02x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # B.T handles the transposition in PyTorch, ensuring accuracy and optimized performance
        return torch.matmul(A, B.T)

# Example usage:
# model = Model()
# A = torch.rand(1024, 4096, device='cuda', dtype=torch.float32)
# B = torch.rand(2048, 4096, device='cuda', dtype=torch.float32)
# output = model(A, B)
