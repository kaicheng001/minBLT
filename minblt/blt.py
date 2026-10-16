"""Byte Latent Transformer: local encoder, causal patch model, local decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from .base import BOS, OFFSET, Block, CrossAttention, RMSNorm, byte_loss, causal_mask, validate_bytes
from .patch import EntropyPatcher, FixedPatcher, decoder_patch_ids, patch_mask

# The first three polynomial bases of Meta's byte_group_hash_function.
HASH_PRIMES = (1000000007, 5915587277, 1500450271)


def byte_hash(tokens, length, buckets, function=0):
    """Hash left-padded n-grams of internal byte IDs, as in Meta's byte hash."""
    # int64 powers and sums wrap before the final table modulo.
    powers = tokens.new_tensor(HASH_PRIMES[function]) ** torch.arange(length, device=tokens.device)
    grams = F.pad(tokens, (length-1, 0)).unfold(1, length, 1)
    return (grams*powers).sum(-1).remainder(buckets)


class HashEmbedding(nn.Module):
    def __init__(self, dim, lengths, buckets, functions=1):
        super().__init__()
        self.lengths, self.buckets = tuple(lengths), buckets
        self.tables = nn.ModuleList([nn.Embedding(buckets, dim) for _ in range(functions) for _ in lengths])

    def forward(self, tokens):
        total = 0
        for i, table in enumerate(self.tables):
            function, n = divmod(i, len(self.lengths))
            total = total + table(byte_hash(tokens, self.lengths[n], self.buckets, function))
        return total


class LocalEncoder(nn.Module):
    def __init__(self, dim, global_dim, layers, heads, window, k, ngrams, buckets, functions, theta):
        super().__init__()
        self.window, self.k = window, k
        self.embedding = nn.Embedding(260, dim)
        self.hash = HashEmbedding(dim, ngrams, buckets, functions)
        self.blocks = nn.ModuleList([Block(dim, heads, theta) for _ in range(layers)])
        self.query_projection = nn.Linear(dim, k*dim, bias=False)
        self.cross_attention = nn.ModuleList([CrossAttention(dim, heads) for _ in range(layers)])
        self.to_global = nn.Linear(k*dim, global_dim, bias=False)

    def forward(self, tokens, patches):
        """Unpadded input: [1,T,D] byte states and [1,P,K,D] patch queries."""
        h = self.embedding(tokens) + self.hash(tokens)
        allow = causal_mask(torch.ones_like(tokens, dtype=torch.bool), self.window)
        ids = torch.tensor(patches.ids, device=tokens.device)
        query_ids = torch.arange(len(patches.starts), device=tokens.device).repeat_interleave(self.k)
        membership = patch_mask(query_ids, ids)
        queries = None
        for block, cross in zip(self.blocks, self.cross_attention):
            h = block(h, allow)
            if queries is None:
                # A patch mean initializes K distinct queries at the first layer.
                pooled = torch.stack([h[0, s:s+n].mean(0)
                                       for s, n in zip(patches.starts, patches.lengths)])[None]
                queries = self.query_projection(pooled).reshape(1, -1, h.shape[-1])
            # The reference local model adds a residual outside CrossAttention too.
            queries = queries + cross(queries, h, membership)
        return h, queries.reshape(1, len(patches.starts), self.k, h.shape[-1])


class GlobalTransformer(nn.Module):
    def __init__(self, dim, layers, heads, theta):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, theta) for _ in range(layers)])

    def forward(self, patches):
        valid = torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
        allow = causal_mask(valid)
        for block in self.blocks:
            patches = block(patches, allow)
        return patches


class LocalDecoder(nn.Module):
    def __init__(self, dim, global_dim, layers, heads, window, k, cross_window, theta):
        super().__init__()
        self.k, self.window, self.cross_window = k, window, cross_window
        self.patch_projection = nn.Linear(global_dim, k*dim, bias=False)
        self.cross_attention = nn.ModuleList([CrossAttention(dim, heads) for _ in range(layers)])
        self.blocks = nn.ModuleList([Block(dim, heads, theta) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)

    def forward(self, byte_states, global_states, patches):
        # Encoder input is z=[BOS]+x. At target t, z[t] is BOS or x[t-1].
        # Removing the singleton encoder patch from the lengths shifts patch
        # conditioning: decoder patch j reads global patch j, not j+1.
        h = byte_states[:, :-1]
        context = self.patch_projection(global_states).reshape(1, -1, h.shape[-1])
        query_ids = torch.tensor(decoder_patch_ids(patches), device=h.device)
        key_ids = torch.arange(global_states.shape[1], device=h.device).repeat_interleave(self.k)
        cross_mask = patch_mask(query_ids, key_ids, self.cross_window)
        local_mask = causal_mask(torch.ones(h.shape[:2], dtype=torch.bool, device=h.device), self.window)
        for cross, block in zip(self.cross_attention, self.blocks):
            h = h + cross(h, context, cross_mask)
            h = block(h, local_mask)
        return self.head(self.norm(h))


class ByteLatentTransformer(nn.Module):
    def __init__(self, local_dim=32, global_dim=96, encoder_layers=2, global_layers=3,
                 decoder_layers=2, heads=4, cross_attn_k=2, window=32,
                 cross_window=None, ngrams=(3, 4, 5, 6, 7, 8), hash_buckets=256,
                 hash_functions=1, rope_theta=10000.0):
        super().__init__()
        self.config = dict(local_dim=local_dim, global_dim=global_dim, encoder_layers=encoder_layers,
                           global_layers=global_layers, decoder_layers=decoder_layers, heads=heads,
                           cross_attn_k=cross_attn_k, window=window, cross_window=cross_window,
                           ngrams=tuple(ngrams), hash_buckets=hash_buckets,
                           hash_functions=hash_functions, rope_theta=rope_theta)
        self.encoder = LocalEncoder(local_dim, global_dim, encoder_layers, heads, window,
                                    cross_attn_k, ngrams, hash_buckets, hash_functions, rope_theta)
        self.global_model = GlobalTransformer(global_dim, global_layers, heads, rope_theta)
        self.decoder = LocalDecoder(local_dim, global_dim, decoder_layers, heads, window,
                                    cross_attn_k, cross_window, rope_theta)

    def forward(self, byte_batch, patcher):
        """[B,T] bytes -> [B,T,256] logits and mean next-byte loss."""
        valid = validate_bytes(byte_batch)
        outputs = []
        # Slice each example so padding cannot enter attention, pooling or hashes.
        for row, mask in zip(byte_batch, valid):
            n = int(mask.sum())
            raw = row[:n]
            patches = patcher(raw.tolist()).with_bos()
            tokens = torch.cat((raw.new_tensor([BOS]), raw+OFFSET))[None]
            byte_states, queries = self.encoder(tokens, patches)
            patch_states = self.encoder.to_global(queries.flatten(-2))
            global_states = self.global_model(patch_states)
            logits = self.decoder(byte_states, global_states, patches)
            outputs.append(F.pad(logits, (0, 0, 0, byte_batch.shape[1]-n)))
        logits = torch.cat(outputs)
        return logits, byte_loss(logits, byte_batch)[0]

    @torch.no_grad()
    def next_logits(self, prefix, patcher, placeholder=0):
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

    def save(self, path, patcher):
        auxiliary = None
        if isinstance(patcher, FixedPatcher):
            config = dict(mode="fixed", patch_size=patcher.patch_size)
        else:
            config = dict(mode="entropy", threshold=patcher.threshold, policy=patcher.policy,
                          max_patch_length=patcher.max_patch_length)
            auxiliary = dict(config=patcher.model.config, state_dict=patcher.model.state_dict())
        torch.save(dict(config=self.config, state_dict=self.state_dict(),
                        patcher=config, entropy=auxiliary), path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a model and its patcher from a .pt checkpoint."""
        from .entropy import EntropyModel

        saved = torch.load(path, map_location=device, weights_only=True)
        model = cls(**saved["config"]).to(device).eval()
        model.load_state_dict(saved["state_dict"])
        config = saved["patcher"].copy()
        if config.pop("mode") == "fixed":
            patcher = FixedPatcher(**config)
        else:
            auxiliary = saved["entropy"]
            entropy_model = EntropyModel(**auxiliary["config"]).to(device)
            entropy_model.load_state_dict(auxiliary["state_dict"])
            patcher = EntropyPatcher(entropy_model, **config)
        return model, patcher
