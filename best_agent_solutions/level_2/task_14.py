# level 2 index 14 agent name: KernelAgent o1 speedup: 2.44x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

#
# In this version, we further parallelize the per-row computation of
# (x @ w^T).sum(dim=1).  Instead of assigning one thread per row (which
# sequentially loops over hidden_size*input_size), we let all threads
# in a block collaboratively compute the dot products and partial sums
# for a single row.  This extra parallelism can improve performance.
#

_fused_src_v2 = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// We fuse the operation:
//   1) M = x @ w^T        (size: [batch_size, hidden_size])
//   2) summation over hidden_size => a single scalar per row
//   3) scale by (scaling_factor / 2)
// into a single kernel.  Each block computes one row of "x" in parallel.

__global__ void fused_matmul_sum_scale_kernel_v2(
    const float* __restrict__ x,        // [batch_size, input_size]
    const float* __restrict__ w,        // [hidden_size, input_size]
    const float  scale_factor,          // scaling_factor / 2
    const int    batch_size,
    const int    input_size,
    const int    hidden_size,
    float* __restrict__ out            // [batch_size]
) {
    // One block per row of x
    int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    // blockDim.x is the number of threads per block.  We use
    // shared memory to:
    //   (1) store the entire row x[row,:] (input_size floats)
    //   (2) store all of w (hidden_size * input_size floats)
    //   (3) store partial sums for final reduction (blockDim.x floats)
    extern __shared__ float shared_mem[];
    float* sW = shared_mem;  // [ hidden_size*input_size ] floats
    float* sX = sW + (hidden_size * input_size); // next chunk: [input_size]
    float* partial = sX + input_size;            // next chunk: [blockDim.x]

    // Number of threads in the block:
    int t  = threadIdx.x;
    int nt = blockDim.x;

    // 1) Load W into shared memory (all threads do a slice)
    int w_size = hidden_size * input_size;
    for (int i = t; i < w_size; i += nt) {
       sW[i] = w[i];
    }

    // 2) Load x[row,:] into shared memory
    for (int i = t; i < input_size; i += nt) {
       sX[i] = x[row * input_size + i];
    }

    __syncthreads();

    // 3) Each thread accumulates a portion of the sum:
    //    sum_{j=0..hidden_size-1} sum_{k=0..input_size-1} sX[k]*sW[j*input_size + k]
    // We'll flatten that double sum into a single loop over total_ops = hidden_size*input_size
    float accum = 0.0f;
    for (int idx = t; idx < w_size; idx += nt) {
        int j = idx / input_size;  // which "hidden" index
        int k = idx % input_size;  // which "input" index
        accum += sX[k] * sW[j * input_size + k];
    }

    // Store partial sum in shared memory
    partial[t] = accum;
    __syncthreads();

    // 4) Parallel reduction of partial sums within the block
    //    Typical tree reduction
    for (int stride = nt / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            partial[t] += partial[t + stride];
        }
        __syncthreads();
    }

    // 5) Thread 0 writes the final scaled result
    if (t == 0) {
        // partial[0] is the sum of all dot products
        float val = partial[0] * scale_factor;
        out[row] = val;
    }
}

torch::Tensor fused_matmul_sum_scale_cuda(
    torch::Tensor x,
    torch::Tensor w,
    double scale_factor
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");

    const auto batch_size  = x.size(0);
    const auto input_size  = x.size(1);
    const auto hidden_size = w.size(0);

    // We'll make an output array of size [batch_size], then reshape to (batch_size, 1).
    auto out = torch::empty({batch_size}, x.options());

    // We'll let each block compute sums for one row of x.
    // We choose a fixed block_size here. 256 is often a decent pick.
    const int block_size = 256;
    const int grid_size  = batch_size;

    // We need shared memory for:
    //    w_size = hidden_size * input_size
    //    + input_size
    //    + block_size (for partial sums)
    int w_size = hidden_size * input_size;
    size_t shmem_bytes = (w_size + input_size + block_size) * sizeof(float);

    fused_matmul_sum_scale_kernel_v2<<<grid_size, block_size, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        static_cast<float>(scale_factor),
        (int)batch_size,
        (int)input_size,
        (int)hidden_size,
        out.data_ptr<float>()
    );

    return out;
}
''';

_fused_decl_v2 = r'''
torch::Tensor fused_matmul_sum_scale_cuda(
    torch::Tensor x,
    torch::Tensor w,
    double scale_factor
);
'''

# Build/Load the extension inline
_fused_module_v2 = load_inline(
    name="fused_matmul_sum_scale_v2",
    cpp_sources=_fused_decl_v2,
    cuda_sources=_fused_src_v2,
    extra_cflags=["-O2"],
    extra_ldflags=[],
    functions=["fused_matmul_sum_scale_cuda"],
    verbose=False
)

class Model(nn.Module):
    """
    Model that performs (x @ weight^T).sum(dim=1) * (scaling_factor/2),
    now in a more parallel fused CUDA kernel to reduce runtime further.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        # Keep the identical initialization as the origin
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused forward pass that:
          1) Computes M = x @ weight^T,
          2) Sums M across columns (dim=1),
          3) Scales by (self.scaling_factor / 2).
        Returns a (batch_size x 1) tensor.
        """
        out_flat = _fused_module_v2.fused_matmul_sum_scale_cuda(
            x,
            self.weight,
            float(self.scaling_factor / 2.0)
        )
        # Reshape to (batch_size, 1) to match original output shape
        return out_flat.view(-1, 1)
