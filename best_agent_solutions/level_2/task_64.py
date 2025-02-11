# level 2 index 64 agent name: KernelAgent o1 speedup: 3.34x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###############################################################################
# Inline CUDA code for rowwise LogSumExp + (2xLeakyReLU) + (2xGELU) (fused)
# with an algebraic simplification: 2xLeakyReLU(negative_slope=α) 
# becomes 1xLeakyReLU(negative_slope=α²).
###############################################################################
# Original sequence of ops for each row was:
#    1) logsumexp
#    2) leaky_relu(..., alpha=0.01)
#    3) leaky_relu(..., alpha=0.01)
#    4) gelu
#    5) gelu
# Because applying leaky_relu twice at the same slope α is equivalent to 
# a single leaky_relu at slope α², we replace the two leaky_relu calls 
# with one. The 2x gelu calls remain unchanged because there's no simple 
# algebraic simplification for gelu(gelu(x)).
#
# The input shape is (B, F).
# The output shape is (B, 1).
###############################################################################

fused_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Simple device function for leaky ReLU
__device__ inline float leaky_relu_f(float x, float negative_slope) {
    return (x >= 0.f) ? x : negative_slope * x;
}

// Approximation or direct formula for GELU
//   GELU(x) = 0.5 * x * (1 + erf( x / sqrt(2) ))
__device__ inline float gelu_f(float x) {
    const float kAlpha = 0.70710678118654752440f; // 1 / sqrt(2)
    float cdf = 0.5f * (1.f + erff(x * kAlpha));
    return x * cdf;
}

// We do a block-wide reduction in two passes:
//   1) find row-wise maximum
//   2) sum of exp(x - max)
// Then out = max + log(sum_of_exp)
// Then apply LeakyReLU (with negative_slope^2), then 2x GELU.

template <unsigned int BlockSize>
__global__ void rowwise_logsumexp_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B,
    const int F,
    const float negative_slope)
{
    // Each block handles one row of size F
    int row = blockIdx.x;
    if (row >= B) return;

    // First pass: find the maximum value in this row
    __shared__ float sdata[BlockSize];
    float thread_max = -FLT_MAX;

    for (int j = threadIdx.x; j < F; j += BlockSize) {
        float val = input[row * F + j];
        thread_max = fmaxf(thread_max, val);
    }

    // Reduce within block to find row max
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];

    // Second pass: compute sum of exp(x - row_max)
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < F; j += BlockSize) {
        float val = input[row * F + j];
        thread_sum += expf(val - row_max);
    }

    // Store partial sums, then reduce
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float row_sum_exp = sdata[0];
    float lse = row_max + logf(row_sum_exp); // logsumexp

    // Algebraic simplification:
    // Instead of leaky_relu(..., alpha) twice, 
    // we do a single leaky_relu(..., alpha^2).
    float neg_slope_squared = negative_slope * negative_slope;
    float out = leaky_relu_f(lse, neg_slope_squared);

    // Still apply gelu twice
    out = gelu_f(out);
    out = gelu_f(out);

    if (threadIdx.x == 0) {
        output[row] = out;
    }
}

torch::Tensor rowwise_logsumexp_fused_cuda(
    torch::Tensor input, 
    float negative_slope)
{
    // input shape: (B, F)
    // output shape: (B, 1)
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must have 2 dims");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");

    const auto B = input.size(0);
    const auto F = input.size(1);

    auto out_options =
        torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device())
        .requires_grad(false);

    // We'll put the result as shape (B, 1), but we'll first allocate (B) and then view.
    auto output = torch::empty({B}, out_options);

    const unsigned int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid(B);

    rowwise_logsumexp_fused_kernel<blockSize>
        <<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            B,
            F,
            negative_slope);

    return output.view({(long)B, 1});
}
"""

fused_cpp_signatures = r"""
torch::Tensor rowwise_logsumexp_fused_cuda(torch::Tensor input, float negative_slope);
"""

# Load/compile our fused kernel
fused_ops = load_inline(
    name="rowwise_logsumexp_fused",
    cpp_sources=fused_cpp_signatures,
    cuda_sources=fused_source,
    functions=["rowwise_logsumexp_fused_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

###############################################################################
# Optimized Model definition with algebraic simplification:
#    double leaky_relu(..., alpha=0.01) -> single leaky_relu(..., alpha=0.01^2)
###############################################################################
class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), 
    followed by LogSumExp(dim=1, keepdim=True),
    then effectively 2x LeakyReLU (now replaced by 1x with slope^2),
    then 2x GELU.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        # Same random initialization as the original linear
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # We'll keep the original "negative_slope" parameter (0.01).
        # Our kernel code does negative_slope^2 internally.
        self.negative_slope = 0.01

    def forward(self, x):
        # Use PyTorch's efficient Linear (GEMM) for in_features -> out_features
        x = self.linear(x)
        # Now call our fused rowwise LogSumExp + 1xLeakyReLU(α^2) + 2xGELU kernel
        x = fused_ops.rowwise_logsumexp_fused_cuda(x, self.negative_slope)
        return x

###############################################################################
# The following remains the same as in the original snippet for testing
###############################################################################
batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
