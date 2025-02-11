# level 1 index 99 agent name: KernelAgent 4o speedup: 1.14x

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Euclidean distance computation
        dist_pos = torch.norm(anchor - positive, p=2, dim=1)
        dist_neg = torch.norm(anchor - negative, p=2, dim=1)
        
        # Compute triplet margin loss directly
        triplet_loss = torch.relu(dist_pos - dist_neg + self.margin)
        
        # Return the mean loss
        return triplet_loss.mean()

# Testing the optimized model
batch_size = 128
input_shape = (4096, )

def get_inputs():
    scale = torch.randn(())
    return [torch.randn(batch_size, *input_shape).cuda() * scale, 
            torch.randn(batch_size, *input_shape).cuda(), 
            torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1.0]  # Default margin

# Instantiate and test the model
model = Model(*get_init_inputs()).cuda()
inputs = get_inputs()
output = model(*inputs)
print(output)
