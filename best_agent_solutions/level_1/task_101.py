# level 1 index 101 agent name: KernelAgent o1 speedup: 9.26x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

################################################################################
# Single-pass CUDA kernel implementation for Hinge Loss
# with warp-based block reductions, inline-compiled via cpp_extension.
################################################################################

hinge_loss_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////
// Warp- and block-level reductions for better performance
////////////////////////////////////////////////////////////////

static __inline__ __device__ float warpReduceSum(float val) {
    // Reduce within a warp using shuffle intrinsics
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];  // up to 1024 threads => 1024/32=32 warps
    int lane = threadIdx.x & 31;         // thread index within warp
    int wid  = threadIdx.x >> 5;         // warp index within block

    // Each warp performs partial reduction
    val = warpReduceSum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;

    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

////////////////////////////////////////////////////////////////
// Single-pass kernel: each block does partial sums of hinge
// and uses atomicAdd to accumulate in a global sum
////////////////////////////////////////////////////////////////

__global__ void hinge_loss_single_pass_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ global_sum,
    const int batch_size,
    const int dim)
{
    int global_size = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;

    // Loop over elements (batch_size * dim) this thread will handle
    while (tid < batch_size * dim) {
        int row = tid / dim;
        float val = predictions[tid] * targets[row];
        float hinge = 1.f - val;
        if (hinge < 0.f) {
            hinge = 0.f;
        }
        local_sum += hinge;
        tid += global_size;
    }

    // Now reduce within the block
    local_sum = blockReduceSum(local_sum);

    // One thread per block does an atomicAdd to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, local_sum);
    }
}

// The main entry point that launches our single-pass kernel and returns a scalar
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are contiguous float tensors on CUDA
    predictions = predictions.contiguous();
    targets     = targets.contiguous();

    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.dtype() == torch::kFloat32, "targets must be float32");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D (batch_size, dim)");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D (batch_size)");

    const int batch_size = predictions.size(0);
    const int dim        = predictions.size(1);
    const int total_size = batch_size * dim;

    // We'll store a single float in device_out
    auto device_out = torch::zeros({1}, predictions.options());

    // Configure kernel launch
    const int block_size = 256;
    const int grid_size  = (total_size + block_size - 1) / block_size;

    // Launch the single-pass hinge loss kernel
    hinge_loss_single_pass_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        device_out.data_ptr<float>(),
        batch_size,
        dim
    );

    // Divide on the device to get the mean
    device_out.div_(static_cast<float>(total_size));

    // Reshape to a scalar (0D tensor)
    auto result = device_out.reshape({});

    return result;
}
"""

hinge_loss_cpp_source = r"""
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Build/load the inline extension
hinge_loss_module = load_inline(
    name="hinge_loss_optimized",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=False
)

################################################################################
# The Model class with the same interface as the original, but using the new
# single-pass kernel with warp-based block reductions.
################################################################################

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks, but now
    using an optimized single-pass kernel for higher performance.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        # Same hinge loss, but much faster thanks to the optimized CUDA kernel
        return hinge_loss_module.hinge_loss_cuda(predictions, targets)


################################################################################
# The same input generation functions as in the original code.
################################################################################

batch_size = 1024
input_shape = (1024,)
dim = 1

def get_inputs():
    # predictions: shape (1024, 1024)
    # targets: shape (1024,)
    return [
        torch.randn(batch_size, *input_shape, device='cuda', dtype=torch.float32),
        (torch.randint(0, 2, (batch_size,), device='cuda', dtype=torch.float32) * 2 - 1)
    ]

def get_init_inputs():
    return []
