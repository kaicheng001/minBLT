# minblt

A small, readable PyTorch Byte Latent Transformer for learning how BLT works.
It models UTF-8 bytes directly, with 256 output classes and no tokenizer download.
Patches are contextual groups of bytes of arbitrary length, not entries in a BPE
vocabulary. This is an educational model, not a reproduction of published scores.

```text
raw bytes -> local causal encoder -> patch cross-attention pooling
                                             |
                              larger causal global Transformer
                                             |
shifted encoder states -> decoder cross-attention -> local causal decoder
                                             |
                                  next-byte logits (256 classes)
```

## Causal indexing

`model(x, patcher)` returns logit `t` for target `x[t]`, not `x[t+1]`.
The decoder receives a learned BOS at position zero and encoder state `t-1`
at position `t`. An encoder state includes its own byte and preceding bytes only.
Patch `j` pools all its bytes, so its global state is visible only to bytes in
patches `k > j`. A learned initial context handles the first patch. Decoder
cross-attention can use all completed patches; it always masks the unfinished
current patch, even when that patch has already been pooled in teacher forcing.
The last partial patch is similarly unavailable until a new patch starts.

Generation recomputes the model on the prefix plus a dummy target byte. The
dummy's value cannot influence its logit or boundary. This intentionally slow
method uses exactly the training masks and supports empty prompts.
Entropy scoring also evaluates each canonical prefix separately: this avoids
float32 rounding differences between full-sequence and prefix-shaped attention
flipping a hard boundary when an entropy score equals the threshold.

Inputs are `[batch, time]` long tensors with bytes `0..255`; optional **right**
padding uses `256`. Each example must contain at least one byte. Examples are
sliced before attention and pooling. Padding never contributes to context or
loss. The internal BOS is not an output class. Loss is mean valid-byte NLL;
`BPB = total NLL / (valid byte count * ln(2))`.

## Patching

The separately trained `TinyEntropyModel` predicts the next byte from the prefix.
For target position `t`, `H[t] = -sum(p * log(p))` uses only `x[:t]` and is
measured in nats. The frozen entropy model is not trained through the hard
boundary decisions.

- Global: start a patch if `H[t] > threshold`.
- Relative: start if `H[t] - H[t-1] > threshold`.
- Both start at zero; equality does not trigger a boundary. Empty inputs have
  no patches. An optional length cap starts a new patch causally. Relative
  patching measures increases, so sustained high entropy need not split.
- Global calibration uses a training window to approximate a desired average
  patch size, reporting the achieved value. Ties, the length cap, and short
  windows can prevent an exact target. The threshold then stays fixed for
  validation and generation. There is no sequence-wide top-k boundary selection.

`FixedPatcher(4)` is a sanity baseline. Patches may split UTF-8 characters;
keep raw bytes for lossless storage and decode with `errors="replace"` for display.
Random training windows reset BOS and patch state. Held-out BPB uses the same
reset convention per window and is not a document-level benchmark measurement.

## Quick start

Python 3.11+ and CPU PyTorch are sufficient. From the repository root:

```bash
python -m venv .venv
# Activate .venv: Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python train.py --stage all --data tests/input.txt --device cpu --steps-entropy 10 --steps-blt 10 --seq-len 64 --batch-size 2
python demo.py --device cpu --prompt "Better than BPE" --max-new-bytes 32
```

The first command trains both models and writes ignored `models/entropy.pt` and
`models/blt.pt`. The latter contains configs, state dictionaries, the frozen
entropy model and the calibrated threshold, so demo needs only that checkpoint.
Checkpoints use `torch.load(..., weights_only=True)`, not pickled model objects.
The short original corpus in `tests/input.txt` needs no download. Ten steps check
the computation; they do not produce fluent text.

Train stages separately with `--stage entropy` and `--stage blt`. Use
`--patch-mode fixed --patch-size 4` for the baseline, or
`--policy relative --threshold 0.01` for entropy increases. `--threshold` overrides
global calibration; otherwise `--target-patch-size 4` controls its target.
`--max-patch-length 16` is the default safety cap. `--seed` controls initialization
and sampling. External corpora are accepted with `--data path/to/text.txt`.

```python
import torch
from minblt import ByteLatentTransformer, FixedPatcher
model, patcher = ByteLatentTransformer(), FixedPatcher(4)
x = torch.tensor([list("A café!".encode("utf-8"))])
logits, loss = model(x, patcher)
```

To use the trained entropy patcher and inspect actual model-based boundaries:

```python
from demo import load_checkpoint
from minblt import visualize_patches
model, patcher = load_checkpoint("models/blt.pt")
raw = b"Better than BPE!"
print(visualize_patches(raw, patcher(raw)))
output = model.generate(raw, patcher, max_new_bytes=32, top_k=40)
print(repr(output))  # Exact bytes; decode only for human-readable display.
```

## Reading the code

- `minblt/base.py`: byte conventions, RMSNorm, explicit masked attention and blocks.
- `minblt/patch.py`: patch metadata, fixed/global/relative boundaries and calibration.
- `minblt/entropy.py`: independently trained causal next-byte entropy Transformer.
- `minblt/model.py`: hash embeddings, local encoder, global model, decoder and sampling.
- `minblt/__init__.py`: the small public interface.
- `train.py`: AdamW training, held-out byte-weighted BPB and checkpoint saving.
- `demo.py`: checkpoint reloading, byte generation and patch statistics.
- `tests/test_minblt.py`: CPU causality, gradients, alignment and round-trip checks.

The defaults use 32-wide local models and a 64-wide, two-layer global model.
Sinusoidal positions replace RoPE. One encoder pooling attention and one decoder
cross-attention replace the paper's repeated blocks and patch head splitting.
Hash embeddings use one length (3) and 256 buckets by default, with deliberate
collisions; set `ngrams=()` to disable them or pass several lengths. Their
polynomial hashes see only current and preceding bytes. The decoder uses all
past patch states rather than the paper's split-state arrangement.

There is no KV cache, fused attention, distributed training, packed document
masking, large-scale data pipeline or pretrained weight compatibility. The model
recomputes unfinished patches during generation and uses simple per-example
loops. These reductions favor inspectable causal mathematics over speed.

## References and licensing

Algorithmic references are the [BLT paper](https://arxiv.org/abs/2412.09871)
(Sections 2 and 3, Figures 2 and 5) and [Meta's official BLT
implementation](https://github.com/facebookresearch/blt), including its patcher,
local models, latent Transformer and decoder alignment. The short files and
direct interfaces are inspired by [Karpathy's minbpe](https://github.com/karpathy/minbpe).
This implementation was written from scratch; substantial source blocks were not
copied from either repository. It is unofficial and cannot load official weights.

The official Meta code is CC-BY-NC-4.0; minbpe is MIT. Those licenses belong to
their respective projects and are not relicensed here. No license grant for the
new code is asserted: the repository owner should choose one before redistribution.
The included workshop corpus is original text written for this repository.
