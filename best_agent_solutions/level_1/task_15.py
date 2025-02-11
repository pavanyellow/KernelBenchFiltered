# level 1 index 15 agent name: KernelAgent 4o speedup: 1.16x

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Directly perform matrix multiplication as the product of two lower triangular matrices is lower triangular.
        C = torch.matmul(A, B)
        return C

M = 4096

def get_inputs():
    A = torch.randn(M, M, dtype=torch.float32, device='cuda')
    B = torch.randn(M, M, dtype=torch.float32, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
