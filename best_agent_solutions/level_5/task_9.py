# level 5 index 9 agent name: KernelAgent Claude 3.5 Sonnet speedup: 1.01x

from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import triton
import triton.language as tl
import collections.abc
from itertools import repeat

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")

def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst

def get_activation_layer(act_type):
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output

def get_norm_layer(norm_layer):
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, **factory_kwargs)
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class TextProjection(nn.Module):
    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, act_layer, frequency_embedding_size=256, max_period=10000, out_size=None, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], head_first: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out

@triton.jit
def triton_norm_and_mlp(SRC, DEST, WEIGHT, ACT, stride_src, stride_dest, stride_w):
    pid = tl.program_id(0)
    rid = tl.program_id(1)
    row = tl.arange(0, 128) + pid * stride_dest[0]
    col = tl.arange(0, 128) + rid * stride_dest[1]
    mask = tl.load(SRC + row[:, None] * stride_src[0] + col, mask=row < stride_src[0])
    mean = mask.mean(axis=1, keepdims=True)
    stddev = tl.sqrt(((mask - mean) ** 2).mean(axis=1, keepdims=True))
    normalized = (mask - mean) / (stddev + 1e-6)
    weights = tl.load(WEIGHT + col)
    scaled = normalized * weights
    activation = triton.ops.relu(scaled) if ACT else scaled
    tl.store(DEST + row[:, None] * stride_dest[0] + col, activation)

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        if isinstance(patch_size, int):
            self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, **factory_kwargs)
        else:
            self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(act_layer(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x

class ModulateDiT(nn.Module):
    def __init__(self, hidden_size: int, factor: int, act_layer: Callable, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

def get_cu_seqlens(text_mask, img_len):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2
    return cu_seqlens

def attention(q, k, v, mode="flash", drop_rate=0, attn_mask=None, causal=False, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, batch_size=1):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
    elif mode == "flash":
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))
        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.randn(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

def parallel_attention(hybrid_seq_parallel_attn, q, k, v, img_q_len, img_kv_len, cu_seqlens_q, cu_seqlens_kv):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:, img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:, img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:, img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    attn2, *_ = _flash_attn_forward(
        q[:, cu_seqlens_q[1]:],
        k[:, cu_seqlens_kv[1]:],
        v[:, cu_seqlens_kv[1]:],
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** (-0.5),
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)
    return attn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.0, use_conv=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Conv2d(kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_channels, hidden_channels, bias=bias[0], **factory_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity())
        self.fc2 = linear_layer(hidden_channels, out_features, bias=bias[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def modulate(x, shift=None, scale=None):
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def apply_gate(x, gate=None, tanh=False):
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)

class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, heads_num, mlp_width_ratio: str = 4.0, mlp_drop_rate: float = 0.0, act_type: str = "silu", qk_norm: bool = False, qk_norm_type: str = "layer", qkv_bias: bool = True, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.self_attn_k_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(in_channels=hidden_size, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop_rate, **factory_kwargs)

        self.adaLN_modulation = nn.Sequential(act_layer(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor = None):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x

class IndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size, heads_num, depth, mlp_width_ratio: float = 4.0, mlp_drop_rate: float = 0.0, act_type: str = "silu", qk_norm: bool = False, qk_norm_type: str = "layer", qkv_bias: bool = True, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.LongTensor, mask: Optional[torch.Tensor] = None):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x

class SingleTokenRefiner(nn.Module):
    def __init__(self, in_channels, hidden_size, heads_num, depth, mlp_width_ratio: float = 4.0, mlp_drop_rate: float = 0.0, act_type: str = "silu", qk_norm: bool = False, qk_norm_type: str = "layer", qkv_bias: bool = True, attn_mode: str = "torch", dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer, **factory_kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.LongTensor, mask: Optional[torch.LongTensor] = None):
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        x = self.individual_token_refiner(x, c, mask)
        return x

class MMDoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, heads_num: int, mlp_width_ratio: float, mlp_act_type: str = "gelu_tanh", qk_norm: bool = True, qk_norm_type: str = "rms", qkv_bias: bool = False, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.img_attn_k_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)

        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.txt_attn_k_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None, max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None, freqs_cis: tuple = None) -> Tuple[torch.Tensor, torch.Tensor]:
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(vec).chunk(6, dim=-1)
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        if not self.hybrid_seq_parallel_attn:
            attn = attention(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=img_k.shape[0])
        else:
            attn = parallel_attention(self.hybrid_seq_parallel_attn, q, k, v, img_q_len=img_q.shape[1], img_kv_len=img_k.shape[1], cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate)

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate)

        return img, txt

class MMSingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, heads_num: int, mlp_width_ratio: float = 4.0, mlp_act_type: str = "gelu_tanh", qk_norm: bool = True, qk_norm_type: str = "rms", qk_scale: float = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())
        self.k_norm = (qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity())

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(self,x: torch.Tensor,vec: torch.Tensor,txt_len: int,cu_seqlens_q: Optional[torch.Tensor] = None,cu_seqlens_kv: Optional[torch.Tensor] = None,max_seqlen_q: Optional[int] = None,max_seqlen_kv: Optional[int] = None,freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        assert cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        attn = attention(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=x.shape[0])

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)

class Model(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, patch_size: list = [1, 2, 2], in_channels: int = 16, out_channels: int = None, hidden_size: int = 2048, heads_num: int = 16, mlp_width_ratio: float = 4.0, mlp_act_type: str = "gelu_tanh", mm_double_blocks_depth: int = 20, mm_single_blocks_depth: int = 40, rope_dim_list: List[int] = [16, 56, 56], qkv_bias: bool = True, qk_norm: bool = True, qk_norm_type: str = "rms", guidance_embed: bool = False, text_projection: str = "single_refiner", use_attention_mask: bool = False, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch.set_default_dtype(torch.bfloat16)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = 1024
        self.text_states_dim_2 = 1024

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)

        if self.text_projection == "linear":
            self.txt_in = TextProjection(self.text_states_dim, self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        self.vector_in = MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)

        self.guidance_in = (TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs) if guidance_embed else None)

        self.double_blocks = nn.ModuleList([MMDoubleStreamBlock(self.hidden_size, self.heads_num, mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type, qk_norm=qk_norm, qk_norm_type=qk_norm_type, qkv_bias=qkv_bias, **factory_kwargs) for _ in range(mm_double_blocks_depth)])

        self.single_blocks = nn.ModuleList([MMSingleStreamBlock(self.hidden_size, self.heads_num, mlp_width_ratio=mlp_width_ratio, mlp_act_type=mlp_act_type, qk_norm=qk_norm, qk_norm_type=qk_norm_type, **factory_kwargs) for _ in range(mm_single_blocks_depth)])

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels, get_activation_layer("silu"), **factory_kwargs)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_states: torch.Tensor = None, text_mask: torch.Tensor = None, text_states_2: Optional[torch.Tensor] = None, freqs_cos: Optional[torch.Tensor] = None, freqs_sin: Optional[torch.Tensor] = None, guidance: torch.Tensor = None, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])

        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)

        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        for _, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis]
            img, txt = block(*double_block_args)

        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin)]
                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def params_count(self):
        counts = {
            "double": sum([sum(p.numel() for p in block.img_attn_qkv.parameters()) + sum(p.numel() for p in block.img_attn_proj.parameters()) + sum(p.numel() for p in block.img_mlp.parameters()) + sum(p.numel() for p in block.txt_attn_qkv.parameters()) + sum(p.numel() for p in block.txt_attn_proj.parameters()) + sum(p.numel() for p in block.txt_mlp.parameters()) for block in self.double_blocks]),
            "single": sum([sum(p.numel() for p in block.linear1.parameters()) + sum(p.numel() for p in block.linear2.parameters()) for block in self.single_blocks]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}

batch_size = 1
C = 3
c_block = 16
T = 3
hw_over_8 = 32
text_states_dim = 1024
text_length = 10

def get_inputs():
    x = torch.randn(batch_size, c_block, T, hw_over_8, hw_over_8) * 10
    t = torch.full((batch_size,), 3.0)
    text_states = torch.randn(batch_size, text_length, text_states_dim)
    text_states_2 = torch.randn(batch_size, text_states_dim)
    text_mask = torch.ones(batch_size, text_length)

    img_seq_len = 3 * 16 * 16
    head_dim = 3072 // 24

    def get_rotary_embedding(seq_len, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = pos * inv_freq.unsqueeze(0)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(freqs).to(x.device), torch.sin(freqs).to(x.device)

    freqs_cos, freqs_sin = get_rotary_embedding(img_seq_len, head_dim)

    return [x, t, text_states, text_mask, text_states_2, freqs_cos, freqs_sin]

def get_init_inputs():
    return []

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    decoder = Model()
    decoder_output = decoder(*get_inputs())

    print("Decoder output shape:", decoder_output.shape)
