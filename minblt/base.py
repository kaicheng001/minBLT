"""Shared byte conventions and small, explicit Transformer building blocks."""

import math

import torch
from torch import nn
from torch.nn import functional as F

BOS = PAD = 256  # Internal only; the prediction head always has 256 classes.


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
    return torch.cat((torch.full_like(x[:, :1], BOS), x[:, :-1]), dim=1)


def byte_loss(logits, targets):
    valid = targets != PAD
    nll = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                          ignore_index=PAD, reduction="sum")
    mean = nll / valid.sum().clamp_min(1)
    return mean, mean / math.log(2)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Attention(nn.Module):
    """[B,Q,D] queries, [B,K,C] context, [B,Q,K] Boolean ALLOW mask."""

    def __init__(self, dim, heads=4, context_dim=None):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must be divisible by head count")
        self.heads = heads
        context_dim = context_dim or dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context, allow):
        b, q, d = x.shape
        def split(t):
            return t.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        scores = split(self.q(x)) @ split(self.k(context)).transpose(-1, -2)
        scores = scores / math.sqrt(d // self.heads)
        # Empty rows (only padding queries) yield zero attention, never NaNs.
        mask = allow[:, None]
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * mask
        values = weights @ split(self.v(context))
        return self.out(values.transpose(1, 2).reshape(b, q, d))


def causal_mask(valid, window=None):
    t = torch.arange(valid.shape[1], device=valid.device)
    distance = t[:, None] - t[None, :]
    mask = distance >= 0
    if window is not None:
        mask &= distance < window
    return mask[None] & valid[:, :, None] & valid[:, None, :]


def positions(x):
    # Analytic sinusoidal positions avoid a learned maximum context length.
    t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
    angles = t * frequencies
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)[..., :x.shape[-1]]


class Block(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attention = Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x, allow):
        h = self.norm1(x)
        x = x + self.attention(h, h, allow)
        return x + self.ff(self.norm2(x))
