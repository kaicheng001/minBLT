"""The small byte language model used to choose patch boundaries."""

import torch
from torch import nn

from .base import Block, RMSNorm, byte_loss, causal_mask, shifted, validate_bytes


def entropy(logits):
    """Shannon entropy in nats."""
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


class EntropyModel(nn.Module):
    def __init__(self, dim=32, layers=1, heads=4, window=128, rope_theta=10000.0):
        super().__init__()
        self.config = dict(dim=dim, layers=layers, heads=heads, window=window, rope_theta=rope_theta)
        self.embedding = nn.Embedding(260, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, rope_theta) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)

    def forward(self, x):
        valid = validate_bytes(x)
        # Replace padding before offsetting; those positions are masked anyway.
        h = self.embedding(shifted(x.masked_fill(~valid, 0)))
        mask = causal_mask(valid, self.config["window"])
        for block in self.blocks:
            h = block(h, mask)
        logits = self.head(self.norm(h))
        return logits, byte_loss(logits, x)[0]

    @torch.no_grad()
    def entropies(self, values):
        """H[t] is the next-byte entropy given only values[:t]."""
        device = self.embedding.weight.device
        if not len(values):
            return torch.empty(0, device=device)
        self.eval()
        x = torch.tensor([list(values)], dtype=torch.long, device=device)
        # Prefix-shaped calls keep rounding identical offline and online at ties.
        return torch.stack([entropy(self(x[:, :t+1])[0][0, -1]) for t in range(len(values))])
