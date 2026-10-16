"""A separate causal byte LM. Its frozen uncertainty determines patch starts."""

import torch
from torch import nn

from .base import Block, RMSNorm, byte_loss, causal_mask, shifted, validate_bytes


def entropy(logits):
    """Shannon distribution entropy in nats, not the surprise of a target byte."""
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


class EntropyModel(nn.Module):
    def __init__(self, dim=32, layers=1, heads=4, window=128, rope_theta=10000.0):
        super().__init__()
        if layers < 1 or (window is not None and window < 1):
            raise ValueError("Entropy model depth and window must be positive")
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
    def entropies(self, values, mode="prefix"):
        """H[t] uses x[:t] only; natural-log Shannon entropy, not surprise of x[t]."""
        device = self.embedding.weight.device
        if not len(values):
            return torch.empty(0, device=device)
        self.eval()
        raw = list(values)
        if mode == "parallel":
            x = torch.tensor([raw], dtype=torch.long, device=device)
            return entropy(self(x)[0][0])
        if mode != "prefix":
            raise ValueError("Entropy mode must be prefix or parallel")
        entropies = []
        for t in range(len(raw)):
            # Canonical prefix shape matters: differently sized float32 matrix
            # products can round differently and flip a hard threshold at a tie.
            # Use exactly the same computation offline and online, including the
            # dummy target, whose value is excluded by the shifted input stream.
            x = torch.tensor([raw[:t] + [0]], dtype=torch.long, device=device)
            logits, _ = self(x)
            entropies.append(entropy(logits[0, -1]))
        return torch.stack(entropies)


TinyEntropyModel = EntropyModel  # Compatibility name; a single implementation.
