# level 1 index 41 agent name: KernelAgent o1 speedup: 1.16x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# We must also provide get_init_inputs() and get_inputs(), so the evaluator can load and call them.
# These match the usage described:
#   init: (int=4, int=2, int=2, int=3, bool=False)
#   input: (tensor(shape=(16, 64, 128), dtype=torch.float32))
def get_init_inputs():
    """
    Return the arguments needed to initialize the custom Model exactly as specified.
    """
    return [4, 2, 2, 3, False]

def get_inputs():
    """
    Return the input(s) needed to call the forward pass.
    We'll generate a (16, 64, 128) tensor of float32.
    The evaluator will seed and compare solutions accordingly.
    """
    return [torch.randn(16, 64, 128, dtype=torch.float32)]

# Below is a more optimized inline CUDA source that still implements a 1D max-pool kernel.
# We do two things to speed things up:
#   1) We specialize for the common case (kernel_size=4, stride=2, padding=2, dilation=3)
#      by fully unrolling the loop, which removes the loop overhead and branch checks.
#   2) We provide a fallback for other cases. In this usage, we only expect the specialized path.
#
# This should run faster than the naive loop-based version.

maxpool1d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Specialized kernel for 1D max pooling with kernel_size=4, stride=2, padding=2, dilation=3.
// Unroll the 4-element max search to remove loop overhead.
__global__ void maxpool1d_kernel_special_4_2_2_3(
    const float* __restrict__ input,    // [N, C, IN_WIDTH]
    float* __restrict__ output,         // [N, C, OUT_WIDTH]
    const int N,
    const int C,
    const int IN_WIDTH,
    const int OUT_WIDTH
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * OUT_WIDTH;
    if (idx >= total_threads) {
        return;
    }
    // Decode (n, c, x_out)
    int n = idx / (C * OUT_WIDTH);
    int r = idx % (C * OUT_WIDTH);
    int c = r / OUT_WIDTH;
    int x_out = r % OUT_WIDTH;

    // For kernel_size=4, stride=2, padding=2, dilation=3:
    //
    //   start = x_out * stride - padding = x_out * 2 - 2
    //   the valid indices in [start, start + 3*dilation], stepping by dilation=3
    //   so indices are: start, start+3, start+6, start+9
    //   i.e., x_out*2 - 2, x_out*2 + 1, x_out*2 + 4, x_out*2 + 7
    //
    // We'll check each for bounds, and pick the max.
    int base = x_out * 2 - 2;

    float maxval = -3.402823e+38F;  // Float min approx

    // index #0
    int idx0 = base;
    if (idx0 >= 0 && idx0 < IN_WIDTH) {
        float val0 = input[n * (C * IN_WIDTH) + c * IN_WIDTH + idx0];
        if (val0 > maxval) {
            maxval = val0;
        }
    }

    // index #1
    int idx1 = base + 3;
    if (idx1 >= 0 && idx1 < IN_WIDTH) {
        float val1 = input[n * (C * IN_WIDTH) + c * IN_WIDTH + idx1];
        if (val1 > maxval) {
            maxval = val1;
        }
    }

    // index #2
    int idx2 = base + 6;
    if (idx2 >= 0 && idx2 < IN_WIDTH) {
        float val2 = input[n * (C * IN_WIDTH) + c * IN_WIDTH + idx2];
        if (val2 > maxval) {
            maxval = val2;
        }
    }

    // index #3
    int idx3 = base + 9;
    if (idx3 >= 0 && idx3 < IN_WIDTH) {
        float val3 = input[n * (C * IN_WIDTH) + c * IN_WIDTH + idx3];
        if (val3 > maxval) {
            maxval = val3;
        }
    }

    // Write the result
    output[n * (C * OUT_WIDTH) + c * OUT_WIDTH + x_out] = maxval;
}

// Fallback kernel for 1D max pooling with arbitrary kernel_size, stride, padding, dilation.
// In this usage, we won't often hit this path, but it ensures correctness if we do.
__global__ void maxpool1d_kernel_fallback(
    const float* __restrict__ input,    // [N, C, IN_WIDTH]
    float* __restrict__ output,         // [N, C, OUT_WIDTH]
    const int N,
    const int C,
    const int IN_WIDTH,
    const int OUT_WIDTH,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * OUT_WIDTH;
    if (idx >= total_threads) {
        return;
    }

    // Decode (n, c, x_out)
    int n = idx / (C * OUT_WIDTH);
    int r = idx % (C * OUT_WIDTH);
    int channel = r / OUT_WIDTH;
    int x_out = r % OUT_WIDTH;

    // Compute start index in the input
    int start = x_out * stride - padding;

    float maxval = -3.402823e+38F; // float minimum approx

    for (int k = 0; k < kernel_size; k++) {
        int cur_idx = start + k * dilation;
        if (cur_idx >= 0 && cur_idx < IN_WIDTH) {
            float val = input[n * (C * IN_WIDTH) + channel * IN_WIDTH + cur_idx];
            if (val > maxval) {
                maxval = val;
            }
        }
    }
    output[n * (C * OUT_WIDTH) + channel * OUT_WIDTH + x_out] = maxval;
}

// Host function that dispatches either the specialized kernel or the fallback kernel.
torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // We expect a 3D tensor: [N, C, W]
    TORCH_CHECK(input.dim() == 3, "Input must be 3D");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto IN_WIDTH = input.size(2);

    // Compute output width according to the usual maxpool1d formula
    // OUT_WIDTH = floor((IN_WIDTH + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    int64_t OUT_WIDTH = (IN_WIDTH + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(OUT_WIDTH > 0, "Output width computed to be <= 0. Check your parameters.");

    // Allocate output
    auto options = input.options();
    auto output = torch::empty({N, C, OUT_WIDTH}, options);

    int64_t total_threads = N * C * OUT_WIDTH;
    const int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // If parameters match (kernel_size=4, stride=2, padding=2, dilation=3),
    // use our specialized unrolled kernel. Otherwise, fallback.
    if (kernel_size == 4 && stride == 2 && padding == 2 && dilation == 3) {
        maxpool1d_kernel_special_4_2_2_3<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            C,
            IN_WIDTH,
            OUT_WIDTH
        );
    } else {
        maxpool1d_kernel_fallback<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            C,
            IN_WIDTH,
            OUT_WIDTH,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }
    return output;
}
"""

# Declare the function signature in C++ so we can call it from Python.
maxpool1d_cpp_source = r"""
torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

# Build the inline module containing our custom MaxPool1d CUDA implementation.
maxpool1d_module = load_inline(
    name="custom_maxpool1d_optimized",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_source,
    functions=["maxpool1d_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class Model(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False
    ):
        """
        We match the original interface exactly. 
        Original code was:
            self.maxpool = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices)
        We store these parameters here for our custom kernel.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices  # Not used, but kept for interface consistency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the input through our custom CUDA kernel for 1D max pooling,
        replicating nn.MaxPool1d(...).
        """
        return maxpool1d_module.maxpool1d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
