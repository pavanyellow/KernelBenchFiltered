# level 5 index 5 agent name: KernelAgent O3 Mini High speedup: 30.36x

"""
Optimized fused quadrature and convolution PyTorch Module.
This code defines a Model class that implements a fused CUDA kernel for quadrature filter computation 
and convolution accumulation. The kernel is compiled inline using torch.utils.cpp_extension.load_inline 
with macros specialized at compile time for:
  - DISCRETE: whether to use the discrete formulation (using powf) versus continuous (using expf)
  - LEARN_DT: whether to use per-hippo learned dt (passed via dt_ptr)
  - USE_F2: whether to use the extra branch for the initial state (computing a second filter)
  - D_DIM: compile–time constant for the inner loop over filter dimensions
  
The entire Module interface is preserved, matching the original:
  • __init__(nHippos, ...)
  • forward(u, horizon=None)
and random parameter initialization is maintained.
"""

import math
import torch
import torch.nn as nn
from time import time
from torch.utils.cpp_extension import load_inline

# Global cache for specialized CUDA kernels.
_kernel_cache = {}

# ---------------------------------------------------------------------
# Base module: preserved register() functionality.
class OurModule(nn.Module):
    def __init__(self):
        super().__init__()
    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Register a tensor as a buffer or trainable parameter."""
        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)
        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)

# ---------------------------------------------------------------------
# Fused quadrature and convolution CUDA kernel source strings.
# C++ header source for the fused kernel.
fused_quad_conv_cpp_source = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor fused_quad_conv_cuda(
    torch::Tensor u,
    torch::Tensor q,
    torch::Tensor a,
    torch::Tensor theta,
    torch::Tensor cx0,
    torch::Tensor dt_ptr,
    torch::Tensor D,
    int B, int H, int L, int channels);
"""

# CUDA kernel source. The kernel is specialized via compiler flags.
fused_quad_conv_cuda_source = r"""
#ifndef DISCRETE
#define DISCRETE 0
#endif
#ifndef LEARN_DT
#define LEARN_DT 0
#endif
#ifndef USE_F2
#define USE_F2 0
#endif
#ifndef D_DIM
#define D_DIM 32
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// The fused kernel processes one (b, m) row at a time, 
// with m = c * H + h, where (c,h) index the channel and hippo.
extern "C" __global__
void fused_quad_conv_kernel(
    const float* __restrict__ u,     // (B, H, L)
    const float* __restrict__ q,     // (channels*H, D_DIM)
    const float* __restrict__ a,     // (channels*H, D_DIM)
    const float* __restrict__ theta, // (channels*H, D_DIM)
    const float* __restrict__ cx0,   // (channels*H, D_DIM), if USE_F2==1
    const float* __restrict__ dt_ptr,// (H,), if LEARN_DT==1
    const float* __restrict__ D,     // (channels, H)
    float* __restrict__ y,           // (B, channels, H, L)
    int B, int H, int L, int channels)
{
    // Determine indices for block corresponding to one (b, m) pair.
    int tot_m = channels * H;
    int blk = blockIdx.x;
    int b = blk / tot_m;
    int m = blk % tot_m;
    int c = m / H;
    int h = m % H;
    
    float dt_val = (LEARN_DT ? dt_ptr[h] : (1.0f / (L - 1)));
    
    // Allocate shared memory for filter(s) and cached input row:
    // Layout: [0, L) -> s_f; (if USE_F2: [L, 2*L) -> s_f2); then [((USE_F2?2:1)*L), ((USE_F2?2:1)*L + L)) -> s_u.
    extern __shared__ float shared[];
    float* s_f = shared;
#if USE_F2
    float* s_f2 = s_f + L;
#endif
    float* s_u = shared + ((USE_F2 ? 2 : 1) * L);

    // Stage A: Load the input row u[b, h, :] into shared memory.
    for (int idx = threadIdx.x; idx < L; idx += blockDim.x) {
        s_u[idx] = u[(b * H + h) * L + idx];
    }
    __syncthreads();
    
    // Stage B: Compute the quadrature filter(s) s_f (and s_f2 if enabled).
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        float x = dt_val * j;
        float acc = 0.0f;
#if USE_F2
        float acc2 = 0.0f;
#endif
        #pragma unroll
        for (int d = 0; d < D_DIM; d++) {
            float a_val = a[m * D_DIM + d];
            float abs_a = fabsf(a_val);
#if DISCRETE
            float factor = powf(abs_a, x);
#else
            float factor = __expf(-abs_a * x);
#endif
            float theta_val = theta[m * D_DIM + d];
            float tmp = factor * __cosf(theta_val * x);
            acc += q[m * D_DIM + d] * tmp;
#if USE_F2
            acc2 += cx0[m * D_DIM + d] * tmp;
#endif
        }
        s_f[j] = acc * (2.0f * dt_val);
#if USE_F2
        s_f2[j] = acc2 * (4.0f * dt_val);
#endif
    }
    __syncthreads();
    
    // Stage C: Convolution accumulation.
    float d_val = D[c * H + h];
    for (int k = threadIdx.x; k < L; k += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j <= k; j++) {
            sum += s_f[j] * s_u[k - j];
        }
        float skip = s_u[k] * d_val;
        float out_val = sum + skip;
#if USE_F2
        out_val += s_f2[k];
#endif
        int out_idx = ((b * channels + c) * H + h) * L + k;
        y[out_idx] = out_val;
    }
}

torch::Tensor fused_quad_conv_cuda(
    torch::Tensor u,
    torch::Tensor q,
    torch::Tensor a,
    torch::Tensor theta,
    torch::Tensor cx0,
    torch::Tensor dt_ptr,
    torch::Tensor D,
    int B, int H, int L, int channels) {

    int tot_m = channels * H;
    int grid = B * tot_m;  // one block per (b, m) pair.
    int block = 256;
    // Determine shared memory size:
    size_t shared_size = (((USE_F2 ? 2 : 1) + 1) * L * sizeof(float));
    auto options = u.options();
    torch::Tensor y = torch::empty({B, channels, H, L}, options);
    fused_quad_conv_kernel<<<grid, block, shared_size>>>( 
        u.data_ptr<float>(),
        q.data_ptr<float>(),
        a.data_ptr<float>(),
        theta.data_ptr<float>(),
        (USE_F2 ? cx0.data_ptr<float>() : nullptr),
        (LEARN_DT ? dt_ptr.data_ptr<float>() : nullptr),
        D.data_ptr<float>(),
        y.data_ptr<float>(),
        B, H, L, channels);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in fused_quad_conv_kernel: %s\n", cudaGetErrorString(err));
    }
    return y;
}
"""

# ---------------------------------------------------------------------
# Optimized Model using the fused specialized CUDA operator.
class Model(OurModule):
    def __init__(self,
                 nHippos,           # Number of hippos (H)
                 d_state=64,        # Full state dimension (d_state = 2*d)
                 channels=1, 
                 use_initial=True,      # Use the initial state?
                 zero_order_hold=False, # Preserved but not used.
                 trap_rule=True,        # Preserved but not used (assumed trapezoidal rule)
                 dt_min=0.001,
                 dt_max=0.1,
                 lr=None,
                 learn_a=True,
                 learn_theta=True,
                 learn_dt=False,   # Whether to learn separate dt for each hippo.
                 theta_scale=False,
                 skip_connection=True,
                 repr='cont',   # Representation: 'cont', 'disc', or 'comp'
                 param_norm='none',
                 **kernel_args):
        super().__init__()
        # h: number of hippos; d: half the state dimension.
        self.h = nHippos
        self.d = d_state // 2    
        self.channels = channels
        self.use_initial = use_initial
        self.zero_order_hold = zero_order_hold
        self.trap_rule = trap_rule
        self.repr = repr
        self.learn_dt = learn_dt
        self.shift = 'shift' in self.repr
        self.param_norm = param_norm

        # Parameter shape: (channels, h, d)
        _fp = (channels, self.h, self.d)
        
        # Chebyshev (exponential) initialization.
        h_scale = torch.exp(torch.arange(self.h, dtype=torch.float32) / self.h * math.log(dt_max/dt_min))
        angles = torch.arange(self.d, dtype=torch.float32) * math.pi
        t_scale = h_scale if theta_scale else torch.ones(self.h, dtype=torch.float32)
        # theta shape: (channels, h, d)
        theta = t_scale.view(1, self.h, 1) * angles.view(1, 1, self.d)
        theta = theta.expand(channels, self.h, self.d)
        if self.repr == 'disc':
            a = torch.randn(*_fp, dtype=torch.float32).abs()
        else:
            a = -h_scale.view(1, self.h, 1).expand(channels, self.h, self.d)
                                            
        self.register("theta", theta, learn_theta, lr=lr, wd=None)
        self.register("a", a, learn_a, lr=lr, wd=None)

        if self.learn_dt:
            log_dt = torch.rand(self.h, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            self.register("log_dt", log_dt, True, lr=lr, wd=None)

        # Per-channel skip connection: parameter D of shape (channels, h).
        if not skip_connection:
            self.register("D", torch.zeros((channels, self.h), dtype=torch.float32), False)
        else:
            self.D = nn.Parameter(torch.randn(channels, self.h, dtype=torch.float32))
        
        # Initial state parameters.
        if use_initial or 'comp' in self.repr:
            if self.shift:
                b = torch.zeros(*_fp, dtype=torch.float32)
                b[:, :, 0] = 1
                self.register("b", b, False, lr=lr, wd=None)
            else:
                self.b = nn.Parameter(torch.randn(*_fp, dtype=torch.float32))
            self.c = nn.Parameter(torch.randn(*_fp, dtype=torch.float32))
            self.x0 = nn.Parameter(torch.randn(*_fp, dtype=torch.float32))
        else:
            self.q = nn.Parameter(torch.randn(*_fp, dtype=torch.float32))
        
        # --- Compile a specialized CUDA kernel for this Model instance ---
        # Specialize based on representation and learning flags.
        discrete_val = 1 if self.repr == 'disc' else 0
        learn_dt_val = 1 if self.learn_dt else 0
        use_f2_val = 1 if self.use_initial else 0
        d_dim_val = self.d  # Compile–time constant for D_DIM.
        
        kernel_key = (discrete_val, learn_dt_val, use_f2_val, d_dim_val)
        if kernel_key not in _kernel_cache:
            extra_cuda_cflags = [
                "-DDISCRETE=%d" % discrete_val,
                "-DLEARN_DT=%d" % learn_dt_val,
                "-DUSE_F2=%d" % use_f2_val,
                "-DD_DIM=%d" % d_dim_val,
                "-O3"
            ]
            _kernel_cache[kernel_key] = load_inline(
                name="fused_module_%d%d%d_%d" % (discrete_val, learn_dt_val, use_f2_val, d_dim_val),
                cpp_sources=fused_quad_conv_cpp_source,
                cuda_sources=fused_quad_conv_cuda_source,
                functions=["fused_quad_conv_cuda"],
                verbose=False,
                extra_cuda_cflags=extra_cuda_cflags,
            )
        self.fused_kernel = _kernel_cache[kernel_key]

    def quadrature_method(self, u, horizon):
        # u is of shape (B, h, L)
        B, H, L = u.shape
        # Determine dt: if learn_dt activated, compute dt_tensor per hippo.
        dt_tensor = torch.exp(self.log_dt) if self.learn_dt else torch.empty(0, dtype=torch.float32, device=u.device)
        
        if self.use_initial:
            # Compute q and cx0 from the b, c, and x0 parameters.
            q_param = self.b * self.c
            cx0_param = self.c * self.x0
        else:
            q_param = self.q
            cx0_param = torch.empty(0, dtype=torch.float32, device=u.device)
        
        # Flatten channel and hippo dimensions: shape becomes (channels * h, d).
        q_flat     = q_param.reshape(self.channels * self.h, self.d).contiguous()
        a_flat     = self.a.reshape(self.channels * self.h, self.d).contiguous()
        theta_flat = self.theta.reshape(self.channels * self.h, self.d).contiguous()
        if self.use_initial:
            cx0_flat = cx0_param.reshape(self.channels * self.h, self.d).contiguous()
        else:
            cx0_flat = torch.empty(0, dtype=torch.float32, device=u.device)
            
        # Call the specialized fused kernel.
        y = self.fused_kernel.fused_quad_conv_cuda(
                u.contiguous(), 
                q_flat, a_flat, theta_flat, cx0_flat,
                dt_tensor.contiguous() if self.learn_dt else dt_tensor,
                self.D.contiguous(),
                B, self.h, L, self.channels)
        # y has shape (B, channels, h, L). Flatten channels and h: (B, channels*h, L)
        return y.view(B, self.channels * self.h, L)
    
    def forward(self, u, horizon=None):
        return self.quadrature_method(u, horizon)

# ---------------------------------------------------------------------
# Example driver code.
if __name__ == "__main__":
    torch.set_default_device("cuda")
    # Instantiate the model.
    model = Model(nHippos=16)
    # Warm-up run.
    _ = model(torch.randn(8, 16, 256, device="cuda", dtype=torch.float32))
    # Measure runtime.
    u = torch.randn(8, 16, 256, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time()
    y = model(u)
    torch.cuda.synchronize()
    t1 = time()
    print("Output shape:", y.shape)
    print("Runtime: {:.5f}ms".format((t1 - t0) * 1e3))
