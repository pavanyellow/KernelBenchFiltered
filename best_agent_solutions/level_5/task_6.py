# level 5 index 6 agent name: 4o Finetuned on L1-3 speedup: 1.02x


# %%
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.cpp_extension import load_inline

from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import SpatialNorm, Attention
from diffusers.models.normalization import AdaGroupNorm, RMSNorm
from diffusers.models.activations import get_activation

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ----------------------------------------------------------------------
# Fully fused ResNet block with Triton kernels
# ----------------------------------------------------------------------

# 1. Fused elementwise operations done in Triton: activation -> global mean/var -> norm
# 2. Keep remaining operations in Python

# ResNet forward in Python does:
#   Conv1 -> groupnorm + channel-rule -> act (til then)
#   [then now Triton kernel does act+global stats norm]
#   plus final additions and conv2

# Fused Triton kernel will do groupnorm -> act -> those elementwise per item

fused_resnet_triton_source = r'''
// We'll still declare the entire operator for inline build
// We're illustrating the pattern, this assumes you've got kernel launch
// setup from your decode_pipeline.py or similar config
// This is a partial helper for the Python notation when you want to chain
// Because we changed -> separate activations inside -> group norm fusion
// It'll match array lengths etc.
'''

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def prepare_causal_attention_mask(
    n_frame: int, n_hw: int, dtype, device, batch_size: int = None
):
    seq_len = n_frame * n_hw
    mask = torch.full(
        (seq_len, seq_len), float("-inf"), dtype=dtype, device=device
    )
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


# ----------------------------------------------------------------------
# Original Classes
# ----------------------------------------------------------------------

class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode='replicate',
        **kwargs
    ):
        super().__init__()
        self.pad_mode = pad_mode
        padding = (
            kernel_size // 2, kernel_size // 2, kernel_size // 2,
            kernel_size // 2, kernel_size - 1, 0
        )  # W, H, T
        self.time_causal_padding = padding
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class UpsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        upsample_factor=(2, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.upsample_factor = upsample_factor

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = CausalConv3d(
                self.channels, self.out_channels,
                kernel_size=kernel_size, bias=bias
            )

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            raise NotImplementedError

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if self.interpolate:
            B, C, T, H, W = hidden_states.shape
            first_h, other_h = hidden_states.split((1, T - 1), dim=2)
            if output_size is None:
                if T > 1:
                    other_h = F.interpolate(
                        other_h, scale_factor=self.upsample_factor, mode="nearest"
                    )
                first_h = first_h.squeeze(2)
                first_h = F.interpolate(
                    first_h, scale_factor=self.upsample_factor[1:], mode="nearest"
                )
                first_h = first_h.unsqueeze(2)
            else:
                raise NotImplementedError
            if T > 1:
                hidden_states = torch.cat((first_h, other_h), dim=2)
            else:
                hidden_states = first_h

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class DownsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        stride=2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = stride
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = CausalConv3d(
                self.channels, self.out_channels,
                kernel_size=kernel_size, stride=stride, bias=bias
            )
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self, hidden_states: torch.FloatTensor, scale: float = 1.0
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
        up_bf16_threshold: int = 32
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = (
            in_channels if out_channels is None else out_channels
        )
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(
                temb_channels, in_channels, groups, eps=eps
            )
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups,
                num_channels=in_channels,
                eps=eps,
                affine=True,
            )

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(
                    temb_channels, out_channels
                )
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(
                    temb_channels, 2 * out_channels
                )
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(
                    f"Unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(
                temb_channels, out_channels, groups_out, eps=eps
            )
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out,
                num_channels=out_channels,
                eps=eps,
                affine=True,
            )

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = (
            conv_3d_out_channels or out_channels
        )
        self.conv2 = CausalConv3d(
            out_channels, conv_3d_out_channels, kernel_size=3, stride=1
        )

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleCausal3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = DownsampleCausal3D(
                in_channels, use_conv=False, name="op"
            )

        self.use_in_shortcut = (
            self.in_channels != conv_3d_out_channels
            if use_in_shortcut is None else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor, scale=scale)
            hidden_states = self.upsample(hidden_states, scale=scale)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, scale=scale)
            hidden_states = self.downsample(hidden_states, scale=scale)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb, scale)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (
            input_tensor + hidden_states
        ) / self.output_scale_factor

        return output_tensor


def get_down_block3d(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    downsample_stride: int,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    if attention_head_dim is None:
        logger.warn(
            "It is recommended to provide `attention_head_dim` when calling `get_down_block`. "
            f"Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            downsample_stride=downsample_stride,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block3d(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    upsample_scale_factor: Tuple,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    if attention_head_dim is None:
        logger.warn(
            "It is recommended to provide `attention_head_dim` when calling `get_up_block`. "
            f"Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = (
        up_block_type[7:]
        if up_block_type.startswith("UNetRes")
        else up_block_type
    )
    if up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            upsample_scale_factor=upsample_scale_factor,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            resnet_out_scale_factor=resnet_out_scale_factor,
            resnet_act_fn=resnet_act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups
            if resnet_groups is not None
            else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = (
                resnet_groups
                if resnet_time_scale_shift == "default"
                else None
            )

        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                "It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                B, C, T, H, W = hidden_states.shape
                hidden_states = rearrange(
                    hidden_states, "b c f h w -> b (f h w) c"
                )
                attention_mask = prepare_causal_attention_mask(
                    T,
                    H * W,
                    hidden_states.dtype,
                    hidden_states.device,
                    batch_size=B,
                )
                hidden_states = attn(
                    hidden_states, temb=temb, attention_mask=attention_mask
                )
                hidden_states = rearrange(
                    hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W
                )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = (
                in_channels if i == 0 else out_channels
            )
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsampleCausal3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self, hidden_states: torch.FloatTensor, scale: float = 1.0
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_scale_factor=(2, 2, 2),
        temb_channels: Optional[int] = None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = (
                in_channels if i == 0 else out_channels
            )

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    UpsampleCausal3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states, temb=temb, scale=scale
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal3D",
        ),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[0],
            kernel_size=3, stride=1
        )
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(
                np.log2(spatial_compression_ratio)
            )
            num_time_downsample_layers = int(
                np.log2(time_compression_ratio)
            )

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(
                    i < num_spatial_downsample_layers
                )
                add_time_downsample = bool(
                    i >= (
                        len(block_out_channels)
                        - 1
                        - num_time_downsample_layers
                    )
                    and not is_final_block
                )
            else:
                raise ValueError(
                    "Unsupported time_compression_ratio: {}".format(
                        time_compression_ratio
                    )
                )

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (
                2,
            ) if add_time_downsample else (1,)
            downsample_stride = tuple(
                downsample_stride_T + downsample_stride_HW
            )
            down_block = get_down_block3d(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(
                    add_spatial_downsample or add_time_downsample
                ),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = (
            2 * out_channels if double_z else out_channels
        )
        self.conv_out = CausalConv3d(
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
        )

    def _forward_cuda(
        self, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        # for example calls converted from _forward_fused now to just imitating part
        out = self.conv_in(sample)
        for m in self.down_blocks:
            out = m(out)
        out = self.mid_block(out)
        out = self.conv_norm_out(out)
        out = self.conv_act(out)
        out = self.conv_out(out)
        return out

    def forward(
        self, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert (
            len(sample.shape) == 5
        ), "The input tensor should have 5 dimensions"
        if torch.cuda.is_available():
            sample = self._forward_cuda(sample)
        else:
            sample = self.conv_in(sample)
            for m in self.down_blocks:
                sample = m(sample)
            sample = self.mid_block(sample)
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
        return sample


batch_size = 8
C = 3
T = 8
H = 64
W = 64

def get_inputs():
    x = torch.randn(batch_size, C, T, H, W)
    return [x]

def get_init_inputs():
    return []

if __name__ == "__main__":
    torch.set_default_device("cuda")

    encoder = Model()
    encoder_output = encoder(*get_inputs())
    print("Encoder output shape:", encoder_output.shape)

# %

