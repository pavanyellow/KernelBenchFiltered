# level 1 index 6 agent name: KernelAgent o1 speedup: 1.00x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #
        # We'll compile the actual matmul logic once in __init__, so
        # each forward call uses the compiled function for speed.
        #
        def matmul_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return torch.matmul(A, B)

        # In case the environment doesn't have torch.compile or backend="inductor", 
        # fall back to the plain function. Otherwise, use the compiled version.
        if hasattr(torch, 'compile'):
            try:
                self.compiled_fn = torch.compile(matmul_impl, backend="inductor")
            except:
                self.compiled_fn = matmul_impl
        else:
            self.compiled_fn = matmul_impl

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Use the compiled matmul (if available) each time this module is called.
        return self.compiled_fn(A, B)
