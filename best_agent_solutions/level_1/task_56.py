# level 1 index 56 agent name: KernelAgent O3 Mini High speedup: 2.25x

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: tuple, 
                 stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), 
                 groups: int = 1, 
                 bias: bool = False):
        super(Model, self).__init__()
        # This optimized kernel only supports groups==1, bias==False, stride=(1,1),
        # padding=(0,0), and dilation=(1,1).
        assert groups == 1, "Only groups==1 is supported"
        assert bias is False, "Only bias==False is supported"
        assert stride == (1, 1), "Only stride=(1,1) is supported"
        assert padding == (0, 0), "Only padding=(0,0) is supported"
        assert dilation == (1, 1), "Only dilation=(1,1) is supported"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (KH, KW)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create the convolution weight (shape: (out_channels, in_channels, KH, KW)).
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Flatten the inner dimensions (in_channels * KH * KW) and pad that inner dimension to the next power-of-two.
        KH, KW = kernel_size
        K_orig = in_channels * KH * KW
        K_pad = 2 ** math.ceil(math.log2(K_orig))
        weight_flat = self.weight.view(out_channels, -1)
        weight_pad = torch.zeros((out_channels, K_pad), dtype=weight_flat.dtype, device=weight_flat.device)
        weight_pad[:, :K_orig] = weight_flat
        self.register_buffer("weight_pad", weight_pad)
        self.K_orig = K_orig
        self.K_pad = K_pad

    @staticmethod
    @triton.jit
    def _fused_conv2d_im2col_dot_kernel(
        input_ptr,         # pointer to input tensor, shape: (B, C, IH, IW)
        weight_ptr,        # pointer to padded weight tensor, shape: (N, K_pad)
        output_ptr,        # pointer to output tensor, shape: (B, N, OH, OW)
        M: tl.constexpr,   # total number of output pixels = B * OH * OW
        N: tl.constexpr,   # number of output channels (N)
        K_orig: tl.constexpr,  # original inner dimension = in_channels * KH * KW
        K_pad: tl.constexpr,   # padded inner dimension (power-of-2 >= K_orig)
        IH: tl.constexpr,  # input height
        IW: tl.constexpr,  # input width
        OH: tl.constexpr,  # output height = IH - KH + 1
        OW: tl.constexpr,  # output width  = IW - KW + 1
        KH: tl.constexpr,  # kernel height
        KW: tl.constexpr,  # kernel width
        BLOCK_M: tl.constexpr,  # number of output pixels per tile along the M axis
        BLOCK_N: tl.constexpr,  # number of output channels per tile along the N axis
        BLOCK_K: tl.constexpr   # inner dimension tile size (set to K_pad)
    ):
        # This kernel fuses two adjacent operations that an unfused implementation would perform separately:
        # (1) converting each output’s receptive field into an “unfolded” (im2col) tile and
        # (2) multiplying that tile with a tile from the padded weight matrix.
        #
        # Compute the number of input channels from K_orig and kernel dimensions.
        C = K_orig // (KH * KW)
        # For a contiguous input tensor (B, C, IH, IW) the stride values are:
        stride_b = C * IH * IW  # jump over one batch element
        stride_h = IW           # jump one row (height)
        stride_w = 1            # jump one column (width)
        
        # Compute output strides for tensor (B, N, OH, OW)
        out_stride_b = N * OH * OW
        out_stride_oc = OH * OW
        out_stride_h = OW
        out_stride_w = 1

        # Identify a 2D tile:
        #   - The first axis (pid_m) covers BLOCK_M flattened output pixels.
        #   - The second axis (pid_n) covers BLOCK_N output channels.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        m = m_offset + tl.arange(0, BLOCK_M)   # shape: (BLOCK_M,)
        n = n_offset + tl.arange(0, BLOCK_N)     # shape: (BLOCK_N,)

        # Unflatten m into (b, oh, ow). For output shape (B, OH, OW):
        b_idx = m // (OH * OW)
        residual = m % (OH * OW)
        oh = residual // OW
        ow = residual % OW

        # For valid convolution each output pixel’s input patch starts at:
        # base = b_idx * (C * IH * IW) + oh * IW + ow.
        base = b_idx * (C * IH * IW) + oh * IW + ow  # shape: (BLOCK_M,)

        # Compute a “delta” offset for the kernel patch:
        # For each k in 0..(BLOCK_K-1), representing flattened (c, kh, kw):
        #   delta[k] = (k // (KH*KW)) * (IH*IW) + ((k % (KH*KW)) // KW) * IW + (k % KW)
        k = tl.arange(0, BLOCK_K)
        mask_k = None if K_orig == BLOCK_K else (k < K_orig)
        delta = ((k // (KH * KW)) * (IH * IW) +
                 ((k % (KH * KW)) // KW) * IW +
                 (k % KW))
        
        # Fuse the im2col step with the dot product:
        # Load the input tile (a_tile) of shape (BLOCK_M, BLOCK_K) using the computed base and delta.
        if mask_k is None:
            a_tile = tl.load(input_ptr + base[:, None] + delta[None, :])
        else:
            a_tile = tl.load(input_ptr + base[:, None] + delta[None, :],
                             mask=mask_k[None, :],
                             other=0.0)
        
        # Load the weight tile (w_tile) of shape (BLOCK_K, BLOCK_N) from the padded weight.
        # The weight tensor is stored in row-major order with shape (N, K_pad).
        w_indices = n[None, :] * K_pad + k[:, None]
        w_tile = tl.load(weight_ptr + w_indices)

        # Compute the dot product: (BLOCK_M x BLOCK_K) dot (BLOCK_K x BLOCK_N)
        acc = tl.dot(a_tile, w_tile)

        # Compute the flat output indices for writing results.
        out_index = (b_idx[:, None] * out_stride_b +
                     n[None, :] * out_stride_oc +
                     oh[:, None] * out_stride_h +
                     ow[:, None] * out_stride_w)
        tl.store(output_ptr + out_index, acc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be contiguous with shape (B, C, IH, IW)
        B, C, IH, IW = x.shape
        assert C == self.in_channels, "Input channel count mismatch!"
        KH, KW = self.kernel_size
        # For valid convolution: OH = IH - KH + 1, OW = IW - KW + 1.
        OH = IH - KH + 1
        OW = IW - KW + 1
        # Flatten the output pixels into one dimension: M = B * OH * OW.
        M = B * OH * OW
        N = self.out_channels

        # Tile sizes (tuned for our fixed input shape).
        BLOCK_M = 128      # number of output pixels per tile
        BLOCK_N = 64       # number of output channels per tile
        BLOCK_K = self.K_pad  # inner dimension tile size (padded to a power-of-2)

        # Allocate the output tensor of shape (B, N, OH, OW)
        out = torch.empty((B, N, OH, OW), device=x.device, dtype=x.dtype)
        
        # Launch a 2D grid:
        #   - The first grid dimension covers the flattened output pixels.
        #   - The second grid dimension covers the output channels.
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        Model._fused_conv2d_im2col_dot_kernel[grid](
            x,                  # input_ptr: shape (B, C, IH, IW)
            self.weight_pad,    # weight_ptr: shape (N, K_pad)
            out,                # output_ptr: shape (B, N, OH, OW)
            M, N, self.K_orig, self.K_pad,
            IH, IW, OH, OW,
            KH, KW,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return out
