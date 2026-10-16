# minBLT

A minimal PyTorch reimplementation of the Byte Latent Transformer (BLT).
The small files, direct interfaces and training example follow Andrej Karpathy's
[minbpe](https://github.com/karpathy/minbpe), including its exact
[Taylor Swift corpus](tests/taylorswift.txt). The adaptation credits take their cue
from [minbpe-pytorch](https://github.com/kuprel/minbpe-pytorch).

The algorithms follow the [BLT paper](https://aclanthology.org/2025.acl-long.453/)
and [Meta's implementation](https://github.com/facebookresearch/blt). This project
is unofficial; it makes no claim to reproduce paper-scale training or weights.
BPE learns a fixed vocabulary of merges. BLT instead models raw bytes and forms
contextual, variable-length patches. A patch is not a vocabulary token.

1. [base.py](minblt/base.py): byte conventions, RMSNorm, RoPE, SwiGLU and attention.
2. [patch.py](minblt/patch.py): fixed/entropy boundaries, masks and calibration.
3. [entropy.py](minblt/entropy.py): a separately trained causal byte entropy model.
4. [blt.py](minblt/blt.py): encoder, patch Transformer, decoder and save/load.
5. [train.py](train.py): train both models, reload a checkpoint, or generate bytes.

## quick start

Use Python 3.11+ with CPU PyTorch. From the repository root:

```bash
python -m pip install -r requirements.txt
python train.py
python train.py --generate --prompt "Better than BPE" --max-new-bytes 32
```

The default runs 20 steps per stage. For a shorter CPU check, use
`python train.py --steps-entropy 10 --steps-blt 10 --device cpu`.

The library also accepts ordinary byte tensors (this example is untrained):

```python
import torch
from minblt import ByteLatentTransformer, FixedPatcher
model, patcher = ByteLatentTransformer(), FixedPatcher(4)
raw = "Hello, café!".encode("utf-8")
logits, loss = model(torch.tensor([list(raw)]), patcher)
```

Reload the model and its frozen patcher from the authoritative `.pt` file:

```python
from minblt import ByteLatentTransformer, visualize_patches
model, patcher, metadata = ByteLatentTransformer.load("models/blt.pt")
raw = b"Better than BPE"
print(visualize_patches(raw, patcher(raw)))
output = model.generate(raw, patcher, max_new_bytes=32)
print(repr(output))  # Keep bytes; UTF-8 replacement decoding is only for display.
model.save("models/copy.pt", patcher, metadata)
```

## how it works

```text
bytes -> local encoder + K patch queries -> global patch Transformer
                                               |
shifted encoder states -> patch cross-attention -> local decoder -> 256 logits
```

`B x T` targets become `B x (P+1) x K x D_local` queries, including singleton BOS.
Concatenate K queries for each global step, then split global outputs into K
decoder keys/values. Cross-attention follows each encoder block and precedes each
decoder block. Logit `t` predicts `x[t]` from encoder state `([BOS]+x)[t]` and the
preceding completed patch. The default cross mask selects that patch exactly;
`cross_window=2` includes its predecessor. [lecture.md](lecture.md) derives the lag.

Frozen next-byte entropy `H[t] = -sum(p * log(p))` uses only `x[:t]`, in nats.
Split on `H[t] > threshold`, or on `H[t]-H[t-1] > threshold` in relative mode,
which also forces payload byte 1 as in Meta's helper. Length caps are causal;
equality does not split. Calibration fits a fixed global threshold on a training
window and reports achieved average length. `mode="prefix"` is canonical;
`mode="parallel"` has tested `2e-6`-nat CPU tolerance, agreeing away from ties.

Generation recomputes prefixes with a causally hidden dummy target. External bytes
are `0..255`, right padding is `256`; internal IDs are Meta's BOE=0, BOS=1, EOS=2,
reserved=3, byte+4. Padding is sliced before attention/hash; the head predicts
only bytes. UTF-8 characters may span patches.

## training

The unmodified minbpe corpus has 185,768 bytes, SHA-256
`c2e39cb822d4ae0caac22152cefc306d466e31217a9c5524e493ad2b76792f57`.
The first 90% trains both models; the final 10% is held out. Evaluate its first
1,024 bytes (`--eval-bytes`) with BOS reset per window.
BPB is `sum(NLL) / (valid byte count * ln(2))`.

`--stage entropy` / `--stage blt` run separately; `--data` accepts another file.
Use `--patch-mode fixed --patch-size 4` or `--policy relative --threshold 0.01`
for comparisons. `--threshold` overrides calibration; `--target-patch-size` sets
its target. Fixed patching shares the BLT class, so a separate `basic.py` adds no algorithm.
AdamW trains on seeded windows; ignored `models/` stores weights,
architecture, patch settings and dataset metadata. Twenty steps do not imply fluency.

The constructor exposes depths, K, windows, n-gram lengths and table sizes.
Defaults use lengths 3..8 and one polynomial hash; `ngrams=()` disables hashes.
Hashes match signed-int64 overflow, zero left padding and final table modulo.

## tests

```bash
pytest -v .
```

Tests check reference mathematics, causality, gradients, reload and real training.

## exercise

Build the components in order with [exercise.md](exercise.md).

## lecture

Trace byte probabilities through the model in [lecture.md](lecture.md).

## limitations

| Component | Reference behavior retained |
| --- | --- |
| Entropy and patching | Trained frozen byte LM, global/relative rules, initial singleton, causal cap |
| Local encoder | Configurable local RoPE blocks, projected K queries, repeated patch cross-attention |
| Global model | Wider causal RMSNorm/RoPE/SwiGLU stack over concatenated patch queries |
| Decoder | Shifted reference patch IDs, K-way split conditions, patch/history mask, repeated cross-attention |
| Byte embeddings | Offset symbols, selectable causal polynomial n-grams with signed overflow |

Paper equations use one cross-attention residual and normalize hash sums; the
pinned Meta code adds a second residual and leaves those sums unscaled. This
implementation follows the paper for those two choices. The fixed baseline uses
the same singleton BOS alignment as dynamic mode rather than static BOE padding.
Widths, hash tables and FFN rounding multiples are small; initialization uses
PyTorch defaults. KV caching, fused/distributed execution, document packing,
serving and official checkpoint loading are omitted. Prefix recomputation is slow.

## references / credits

Inspected revisions: [minbpe `1acefe8`](https://github.com/karpathy/minbpe/tree/1acefe89412b20245db5a22d2a02001e547dc602),
[minbpe-pytorch `6917201`](https://github.com/kuprel/minbpe-pytorch/tree/6917201135258f37eb6e4925f8fef6bb2c79244d),
[Meta BLT `9774ed4`](https://github.com/facebookresearch/blt/tree/9774ed4fcc78313f9f218295f3d7e4decdadf2ae).
The [archival paper](https://aclanthology.org/2025.acl-long.453/) (Sections 2–3,
Figures 2 and 4, Appendices E–F) resolves algorithm definitions; the
[arXiv version](https://arxiv.org/abs/2412.09871) has different figure numbering.
The Python code is independently written, with no vendored upstream modules.
The projects and authors named here do not endorse MiniBLT.

## License

The new MiniBLT software is MIT licensed; see [LICENSE](LICENSE).
Upstream minbpe and minbpe-pytorch software is MIT; Meta BLT code is
[CC BY-NC 4.0](https://github.com/facebookresearch/blt/blob/9774ed4fcc78313f9f218295f3d7e4decdadf2ae/LICENSE).
The independently written implementation does not relicense those projects.

The corpus is separately attributed to [Wikipedia's Taylor Swift contributors](https://en.wikipedia.org/wiki/Taylor_Swift)
([history](https://en.wikipedia.org/w/index.php?title=Taylor_Swift&action=history)),
copied by minbpe on February 16, 2024 and redistributed here byte-for-byte under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as its preserved
footer states. Its exact source is the [pinned minbpe data file](https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/tests/taylorswift.txt).
The software's MIT license does not relicense Wikipedia text.
