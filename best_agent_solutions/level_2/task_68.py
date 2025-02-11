# level 2 index 68 agent name: KernelAgent Claude 3.5 Sonnet speedup: 2.72x

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Thread block handles multiple rows, each thread handles multiple columns
#define BLOCK_ROWS 16
#define THREAD_COLS 4
#define BLOCK_SIZE 256

__global__ void fused_linear_nrelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    __shared__ float weight_shared[32][32];  // Cache portion of weight matrix
    
    const int tid = threadIdx.x;
    const int batch_start = blockIdx.x * BLOCK_ROWS;
    const int out_start = threadIdx.x * THREAD_COLS;
    
    // Load bias values for this thread's output features
    float thread_bias[THREAD_COLS];
    #pragma unroll
    for (int j = 0; j < THREAD_COLS; j++) {
        const int out_idx = out_start + j;
        if (out_idx < out_features) {
            thread_bias[j] = bias[out_idx];
        }
    }
    
    // Process BLOCK_ROWS input rows
    for (int batch_idx = batch_start; batch_idx < min(batch_start + BLOCK_ROWS, batch_size); batch_idx++) {
        float results[THREAD_COLS] = {0.0f};
        
        // Compute matrix multiplication in tiles
        for (int tile = 0; tile < in_features; tile += 32) {
            // Collaboratively load weight tile into shared memory
            if (tid < 32) {
                for (int i = 0; i < 32; i++) {
                    const int w_row = tid;
                    const int w_col = tile + i;
                    if (w_row < out_features && w_col < in_features) {
                        weight_shared[w_row][i] = weight[w_row * in_features + w_col];
                    }
                }
            }
            __syncthreads();
            
            // Compute partial results for this tile
            #pragma unroll
            for (int k = 0; k < 32 && tile + k < in_features; k++) {
                const float in_val = input[batch_idx * in_features + tile + k];
                #pragma unroll
                for (int j = 0; j < THREAD_COLS; j++) {
                    const int out_idx = out_start + j;
                    if (out_idx < out_features) {
                        results[j] += in_val * weight_shared[out_idx][k];
                    }
                }
            }
            __syncthreads();
        }
        
        // Write results with negative ReLU
        #pragma unroll
        for (int j = 0; j < THREAD_COLS; j++) {
            const int out_idx = out_start + j;
            if (out_idx < out_features) {
                const float final = results[j] + thread_bias[j];
                output[batch_idx * out_features + out_idx] = (final < 0.0f) ? final : 0.0f;
            }
        }
    }
}

torch::Tensor fused_linear_nrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int blocks = (batch_size + BLOCK_ROWS - 1) / BLOCK_ROWS;
    
    fused_linear_nrelu_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_nrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

fused_module = load_inline(
    name='fused_linear_nrelu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_linear_nrelu_cuda'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        # Pre-subtract constant from bias at initialization
        with torch.no_grad():
            self.linear.bias.sub_(self.constant)

    def forward(self, x):
        return fused_module.fused_linear_nrelu_cuda(
            x,
            self.linear.weight,
            self.linear.bias
        )

batch_size = 128
in_features = 10
out_features = 5
constant = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]
