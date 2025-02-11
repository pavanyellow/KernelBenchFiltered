# level 1 index 11 agent name: KernelAgent 4o speedup: 1.10x

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        # Use batch multiplication and reshape the tensors accordingly
        b, i, j, l = A.shape
        l, k = B.shape
        
        # Reshape A to match with bmm requirement, merging i*j as batch
        A_reshaped = A.view(b * i, j, l)  # Shape: (b*i, j, l)
        
        # Make B a batch-friendly matrix
        B_expanded = B.expand(b * i, l, k)  # Shape: (b*i, l, k)
        
        # Perform batch matrix multiplication
        C_reshaped = torch.bmm(A_reshaped, B_expanded)  # Shape: (b*i, j, k)

        # Reshape back to the desired output shape
        C = C_reshaped.view(b, i, j, k)  # Shape: (b, i, j, k)

        return C

# Test code
b = 16
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.randn(b, i, j, l, device='cuda', dtype=torch.float32)
    B = torch.randn(l, k, device='cuda', dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
