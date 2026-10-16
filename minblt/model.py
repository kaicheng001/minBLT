"""Trace the byte encoder -> patch Transformer -> shifted byte decoder here."""

import torch
from torch import nn
from torch.nn import functional as F

from .base import Attention, Block, RMSNorm, byte_loss, causal_mask, positions, validate_bytes


class HashEmbedding(nn.Module):
    """Small polynomial rolling hashes; collisions intentionally share embeddings."""

    def __init__(self, dim, lengths=(3,), buckets=256):
        super().__init__()
        if buckets < 1 or any(n < 1 for n in lengths):
            raise ValueError("Hash bucket count and n-gram lengths must be positive")
        self.lengths, self.buckets = tuple(lengths), buckets
        self.tables = nn.ModuleList([nn.Embedding(buckets, dim) for _ in lengths])

    def forward(self, x):
        result = 0
        for n, table in zip(self.lengths, self.tables):
            hashes = torch.zeros_like(x)
            # h[t] = sum(x[t-i] * 257**i) mod buckets. Reduce each term
            # before multiplication, avoiding integer overflow for long n-grams.
            for i in range(min(n, x.shape[1])):
                hashes[:, i:] = (hashes[:, i:] + x[:, :x.shape[1]-i]
                                  * pow(257, i, self.buckets)) % self.buckets
            embedding = table(hashes)
            valid = torch.arange(x.shape[1], device=x.device) >= n-1
            result = result + embedding * valid[None, :, None]
        return result


class LocalEncoder(nn.Module):
    def __init__(self, dim, global_dim, heads, window, ngrams, hash_buckets):
        super().__init__()
        self.embedding = nn.Embedding(256, dim)
        self.hash = HashEmbedding(dim, ngrams, hash_buckets)
        self.window = window
        self.block = Block(dim, heads)
        self.norm = RMSNorm(dim)
        self.pool_attention = Attention(dim, heads)
        self.to_global = nn.Linear(dim, global_dim)  # Explicit width conversion.

    def forward(self, x, patches):
        """One unpadded sequence: return [1,T,D_local], [1,P,D_global]."""
        h = (self.embedding(x) + self.hash(x)) / (1+len(self.hash.lengths))
        h = h + positions(h)
        valid = torch.ones_like(x, dtype=torch.bool)
        h = self.block(h, causal_mask(valid, self.window))
        queries = torch.stack([h[0, s:s+n].mean(0)
                               for s, n in zip(patches.starts, patches.lengths)])[None]
        ids = torch.tensor(patches.ids, device=x.device)
        membership = torch.arange(len(patches.starts), device=x.device)[:, None] == ids[None]
        pooled = queries + self.pool_attention(self.norm(queries), self.norm(h), membership[None])
        return h, self.to_global(pooled)


class GlobalTransformer(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(dim)

    def forward(self, patches):
        h = patches + positions(patches)
        valid = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for block in self.blocks:
            h = block(h, causal_mask(valid))
        return self.norm(h)


class LocalDecoder(nn.Module):
    def __init__(self, dim, global_dim, heads, window):
        super().__init__()
        self.bos = nn.Parameter(torch.randn(1, 1, dim)*0.02)
        self.initial_context = nn.Parameter(torch.randn(1, 1, global_dim)*0.02)
        self.query_norm, self.context_norm = RMSNorm(dim), RMSNorm(global_dim)
        self.cross_attention = Attention(dim, heads, global_dim)
        self.block, self.norm = Block(dim, heads), RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)
        self.window = window

    def forward(self, byte_states, global_states, patches):
        # Decoder position t receives encoder state t-1, never state t.
        h = torch.cat((self.bos, byte_states[:, :-1]), dim=1)
        h = h + positions(h)
        context = torch.cat((self.initial_context, global_states), dim=1)
        ids = torch.tensor(patches.ids, device=h.device)
        # Context index 0 is learned initial context. Index j+1 is patch j.
        # For bytes in patch k, allow only 0 and completed patches j < k.
        allow = torch.arange(context.shape[1], device=h.device)[None] <= ids[:, None]
        h = h + self.cross_attention(self.query_norm(h), self.context_norm(context), allow[None])
        valid = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        h = self.block(h, causal_mask(valid, self.window))
        return self.head(self.norm(h))


class ByteLatentTransformer(nn.Module):
    def __init__(self, local_dim=32, global_dim=64, global_layers=2, heads=4,
                 window=32, ngrams=(3,), hash_buckets=256):
        super().__init__()
        self.config = dict(local_dim=local_dim, global_dim=global_dim,
                           global_layers=global_layers, heads=heads, window=window,
                           ngrams=tuple(ngrams), hash_buckets=hash_buckets)
        self.encoder = LocalEncoder(local_dim, global_dim, heads, window, ngrams, hash_buckets)
        self.global_model = GlobalTransformer(global_dim, global_layers, heads)
        self.decoder = LocalDecoder(local_dim, global_dim, heads, window)

    def forward(self, byte_batch, patcher):
        """[B,T] targets -> [B,T,256] logits, valid-byte mean cross-entropy.

        Slice right padding before any attention. Small sequence loops keep variable
        patch counts transparent, and padding cannot enter context or pooling.
        """
        valid = validate_bytes(byte_batch)
        outputs = []
        for row, mask in zip(byte_batch, valid):
            n = int(mask.sum())
            x = row[:n][None]
            patches = patcher(x[0].tolist())
            byte_states, patch_states = self.encoder(x, patches)
            global_states = self.global_model(patch_states)
            logits = self.decoder(byte_states, global_states, patches)
            outputs.append(F.pad(logits, (0, 0, 0, byte_batch.shape[1]-n)))
        logits = torch.cat(outputs)
        return logits, byte_loss(logits, byte_batch)[0]

    @torch.no_grad()
    def next_logits(self, prefix, patcher, placeholder=0):
        # Target placeholder is masked from its own logit and entropy estimate.
        x = torch.tensor([list(prefix)+[placeholder]], dtype=torch.long,
                         device=next(self.parameters()).device)
        return self(x, patcher)[0][0, -1]

    @torch.no_grad()
    def generate(self, prompt_bytes, patcher, max_new_bytes=32, temperature=1.0, top_k=None):
        if max_new_bytes < 0 or temperature <= 0 or (top_k is not None and not 1 <= top_k <= 256):
            raise ValueError("Expected nonnegative byte count, positive temperature and top-k in 1..256")
        was_training = self.training
        self.eval()
        output = bytearray(prompt_bytes)
        try:
            for _ in range(max_new_bytes):
                logits = self.next_logits(output, patcher) / temperature
                if top_k is not None:
                    logits = logits.masked_fill(logits < logits.topk(top_k).values[-1], -float("inf"))
                output.append(int(torch.multinomial(logits.softmax(-1), 1)))
        finally:
            self.train(was_training)
        return bytes(output)
