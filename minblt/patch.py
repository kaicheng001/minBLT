"""Incremental byte boundaries; patch IDs are positions, never vocabulary IDs."""

from dataclasses import dataclass
import torch


def patch_mask(query_ids, key_ids, window=None):
    """Reference cross-attention ALLOW mask [1,Q,K], before K-way expansion."""
    distance = query_ids[:, None] - key_ids[None, :]
    if window is None:
        return (distance == 0)[None]
    if window < 1:
        raise ValueError("Cross-attention window must be positive")
    return ((distance >= 0) & (distance < window))[None]


def decoder_patch_ids(patches):
    # Drop the singleton initial encoder patch, as in Meta's lag correction.
    if not patches.lengths or patches.lengths[0] != 1:
        raise ValueError("Decoder alignment requires a singleton initial encoder patch")
    return [j for j, n in enumerate(patches.lengths[1:]) for _ in range(n)]


@dataclass
class Patches:
    starts: list[int]
    lengths: list[int]
    ids: list[int]

    @classmethod
    def build(cls, starts, size):
        starts = list(starts)
        if size == 0 and not starts:
            return cls([], [], [])
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("Patch starts must be increasing and begin at zero")
        ends = starts[1:] + [size]
        lengths = [end-start for start, end in zip(starts, ends)]
        if min(lengths) <= 0:
            raise ValueError("Patches must be nonempty and inside the byte sequence")
        return cls(starts, lengths, [j for j, n in enumerate(lengths) for _ in range(n)])

    @property
    def average(self):
        return len(self.ids) / len(self.starts) if self.starts else 0.0

    def with_bos(self):
        """Payload patches -> encoder patches for [BOS] + payload bytes."""
        return Patches.build([0] + [s+1 for s in self.starts], len(self.ids)+1)


class FixedPatcher:
    def __init__(self, patch_size=4):
        if patch_size < 1:
            raise ValueError("Patch size must be positive")
        self.patch_size = patch_size

    def __call__(self, values):
        return Patches.build(range(0, len(values), self.patch_size), len(values))


class EntropyPatcher:
    def __init__(self, model, threshold, policy="global", max_patch_length=None):
        assert policy in ("global", "relative")
        self.model, self.threshold = model, float(threshold)
        self.policy, self.max_patch_length = policy, max_patch_length
        if model is not None:
            model.eval().requires_grad_(False)
            for parameter in model.parameters():
                parameter.grad = None

    def from_entropies(self, entropies):
        h = [float(v) for v in entropies]
        starts = [0] if h else []
        for t in range(1, len(h)):
            score = h[t] if self.policy == "global" else h[t] - h[t-1]
            capped = self.max_patch_length is not None and t-starts[-1] >= self.max_patch_length
            # Meta's relative policy forces the first post-BOS candidate.
            initial_relative = self.policy == "relative" and t == 1
            if score > self.threshold or capped or initial_relative:
                starts.append(t)
        return Patches.build(starts, len(h))

    def __call__(self, values):
        return self.from_entropies(self.model.entropies(values).tolist())

    def calibrate(self, values, target_patch_size=4):
        """Choose a global threshold for an approximate average patch length."""
        assert self.policy == "global" and target_patch_size >= 1
        h = self.model.entropies(values).tolist()
        low, high = min(h)-1e-6, max(h)+1e-6
        best = (float("inf"), self.threshold)
        for _ in range(40):
            self.threshold = (low+high)/2
            average = self.from_entropies(h).average
            best = min(best, (abs(average-target_patch_size), self.threshold))
            if average < target_patch_size:
                low = self.threshold
            else:
                high = self.threshold
        self.threshold = best[1]
        return self.from_entropies(h).average


def visualize_patches(values, patches):
    raw = bytes(values)
    pieces = [repr(raw[s:s+n]) for s, n in zip(patches.starts, patches.lengths)]
    return ("patches: " + " ".join("["+p+"]" for p in pieces)
            + f"\nlengths: {patches.lengths}\ncount: {len(pieces)}, average: {patches.average:.2f}")
