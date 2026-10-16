"""A separate causal byte LM. Its frozen uncertainty determines patch starts."""

import torch
from torch import nn

from .base import Block, RMSNorm, byte_loss, causal_mask, positions, shifted, validate_bytes


class TinyEntropyModel(nn.Module):
    def __init__(self, dim=32, layers=1, heads=4, window=128):
        super().__init__()
        self.config = dict(dim=dim, layers=layers, heads=heads, window=window)
        self.embedding = nn.Embedding(257, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)

    def forward(self, x):
        valid = validate_bytes(x)
        h = self.embedding(shifted(x))
        h = h + positions(h)
        mask = causal_mask(valid, self.config["window"])
        for block in self.blocks:
            h = block(h, mask)
        logits = self.head(self.norm(h))
        return logits, byte_loss(logits, x)[0]

    @torch.no_grad()
    def entropies(self, values):
        """H[t] uses x[:t] only; natural-log Shannon entropy, not surprise of x[t]."""
        device = self.embedding.weight.device
        if not len(values):
            return torch.empty(0, device=device)
        self.eval()
        raw = list(values)
        entropies = []
        for t in range(len(raw)):
            # Canonical prefix shape matters: differently sized float32 matrix
            # products can round differently and flip a hard threshold at a tie.
            # Use exactly the same computation offline and online, including the
            # dummy target, whose value is excluded by the shifted input stream.
            x = torch.tensor([raw[:t] + [0]], dtype=torch.long, device=device)
            logits, _ = self(x)
            logp = logits[0, -1].log_softmax(-1)
            entropies.append(-(logp.exp() * logp).sum())
        return torch.stack(entropies)
