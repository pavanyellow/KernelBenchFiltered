# level 1 index 96 agent name: KernelAgent 4o speedup: 1.44x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel_optimized(const float* preds, const float* targets, float* losses, int size, float beta) {
    __shared__ float shared_loss[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    
    if (idx < size) {
        float delta = preds[idx] - targets[idx];
        float abs_delta = fabs(delta);
        float squared_loss = 0.5f * delta * delta / beta;
        float linear_loss = abs_delta - 0.5f * beta;
        local_loss = abs_delta < beta ? squared_loss : linear_loss;
    }

    // Store local_loss in shared memory
    shared_loss[threadIdx.x] = (idx < size) ? local_loss : 0.0f;
    __syncthreads();
    
    // Reduce shared_loss within the block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_loss[threadIdx.x] += shared_loss[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Write result from each block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(losses, shared_loss[0] / size);
    }
}

torch::Tensor smooth_l1_loss_cuda_optimized(torch::Tensor preds, torch::Tensor targets, const float beta) {
    int size = preds.numel();
    auto losses = torch::zeros({1}, preds.options());

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    smooth_l1_loss_kernel_optimized<<<blocks, threads>>>(preds.data_ptr<float>(), targets.data_ptr<float>(), 
                                                         losses.data_ptr<float>(), size, beta);

    return losses[0];
}
"""

# Define the necessary C++ signatures to match the CUDA definition
cpp_source = "torch::Tensor smooth_l1_loss_cuda_optimized(torch::Tensor preds, torch::Tensor targets, float beta);"

# Compile the inline CUDA code
smooth_l1_loss_native_optimized = load_inline(
    name='smooth_l1_loss_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['smooth_l1_loss_cuda_optimized'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        # Ensure inputs are contiguous
        if not predictions.is_contiguous():
            predictions = predictions.contiguous()
        if not targets.is_contiguous():
            targets = targets.contiguous()

        # Call the optimized CUDA function to compute the smooth L1 loss
        loss = smooth_l1_loss_native_optimized.smooth_l1_loss_cuda_optimized(predictions, targets, 1.0)
        
        return loss
