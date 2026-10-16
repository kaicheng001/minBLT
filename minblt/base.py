"""Shared byte conventions and small, explicit Transformer building blocks."""

import math

import torch
from torch import nn
from torch.nn import functional as F

# Public tensors hold bytes 0..255 and right padding 256. Internally use Meta's
# offset-byte arrangement: BOE=0, BOS=1, EOS=2, reserved=3, bytes=4..259.
PAD = 256
BOE, BOS, EOS, OFFSET = 0, 1, 2, 4


def encode(text):
    return list(text.encode("utf-8"))


def decode(values):
    return bytes(values).decode("utf-8", errors="replace")


def validate_bytes(x):
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Expected a nonempty [batch, time] byte tensor")
    if x.dtype != torch.long or ((x < 0) | (x > PAD)).any():
        raise ValueError("Expected integer bytes 0..255 and right padding 256")
    valid = x != PAD
    if (~valid[:, 0]).any() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("Each sequence must contain bytes followed by optional padding")
    return valid


def shifted(x):
    # Target x[t] is predicted from BOS, x[0], ..., x[t-1].
    return torch.cat((torch.full_like(x[:, :1], BOS), x[:, :-1] + OFFSET), dim=1)


def byte_loss(logits, targets):
    valid = targets != PAD
    nll = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                          ignore_index=PAD, reduction="sum")
    mean = nll / valid.sum().clamp_min(1)
    return mean, mean / math.log(2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope(x, theta=10000.0):
    """Rotate adjacent coordinate pairs in [B,H,T,D], as in Meta's RoPE path."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float()/dim)
    angles = torch.arange(x.shape[-2], device=x.device).float()[:, None] * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """[B,Q,D] queries, [B,K,C] context, [B,Q,K] Boolean ALLOW mask."""

    def __init__(self, dim, heads=4, context_dim=None, rotary=False, rope_theta=10000.0):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must be divisible by head count")
        self.heads = heads
        self.rotary, self.rope_theta = rotary, rope_theta
        if rotary and (dim // heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        context_dim = context_dim or dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context, allow):
        b, q, d = x.shape
        def split(t):
            return t.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        query, key, value = split(self.q(x)), split(self.k(context)), split(self.v(context))
        if self.rotary:
            query, key = rope(query, self.rope_theta), rope(key, self.rope_theta)
        # SDPA Boolean True means allow. Give empty rows a temporary finite
        # softmax, then zero them: older CPU SDPA versions can return NaNs for
        # all-masked rows. This also keeps padding-query behavior explicit.
        active = allow.any(-1, keepdim=True)
        safe_allow = allow | ~active
        values = F.scaled_dot_product_attention(query, key, value, attn_mask=safe_allow[:, None])
        values = values * active[:, None]
        return self.out(values.transpose(1, 2).reshape(b, q, d))


def causal_mask(valid, window=None):
    if window is not None and window < 1:
        raise ValueError("Attention window must be positive")
    t = torch.arange(valid.shape[1], device=valid.device)
    distance = t[:, None] - t[None, :]
    mask = distance >= 0
    if window is not None:
        mask &= distance < window
    return mask[None] & valid[:, :, None] & valid[:, None, :]


class FeedForward(nn.Module):
    """Bias-free SwiGLU, with the reference 8D/3 hidden-width rule."""

    def __init__(self, dim, multiple_of=8):
        super().__init__()
        hidden = multiple_of * math.ceil(int(8*dim/3) / multiple_of)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class CrossAttention(nn.Module):
    """Normalized patch cross-attention, without RoPE; one residual per paper."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.query_norm, self.context_norm = RMSNorm(dim), RMSNorm(dim)
        self.attention = Attention(dim, heads)

    def forward(self, x, context, allow):
        return x + self.attention(self.query_norm(x), self.context_norm(context), allow)


class Block(nn.Module):
    def __init__(self, dim, heads=4, rope_theta=10000.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attention = Attention(dim, heads, rotary=True, rope_theta=rope_theta)
        self.ff = FeedForward(dim)

    def forward(self, x, allow):
        h = self.norm1(x)
        x = x + self.attention(h, h, allow)
        return x + self.ff(self.norm2(x))
