# minblt

A small PyTorch implementation of the Byte Latent Transformer (BLT), inspired by
Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe). It follows minbpe's
short-file style and uses the same [training text](tests/taylorswift.txt), with the
upstream acknowledgment style of [minbpe-pytorch](https://github.com/kuprel/minbpe-pytorch).

The model follows the [BLT paper](https://aclanthology.org/2025.acl-long.453/) and
[Meta's implementation](https://github.com/facebookresearch/blt). It reads UTF-8
bytes directly. A separate byte language model estimates next-byte entropy, which
determines where patches begin. Unlike BPE tokens, patches have no fixed vocabulary.

```text
bytes -> local encoder -> patch cross-attention -> global Transformer
            |                                          |
            +-> shifted byte states -> local decoder <-+
```

The encoder constructs several query states per patch. The global Transformer
processes their concatenation, and the decoder attends to the preceding completed
patch to predict individual bytes. The Transformer blocks use causal attention,
RoPE and SwiGLU; the encoder also adds byte n-gram hash embeddings.

The files are:

1. [base.py](minblt/base.py): byte helpers, normalization, attention and Transformer blocks.
2. [patch.py](minblt/patch.py): fixed and entropy patching, calibration and patch masks.
3. [entropy.py](minblt/entropy.py): the small next-byte entropy model.
4. [blt.py](minblt/blt.py): the encoder, global Transformer, decoder and sampling.
5. [train.py](train.py): training, checkpoint loading and generation.

For a closer walkthrough, see [lecture.md](lecture.md) and [exercise.md](exercise.md).

## quick start

Python 3.11+ and PyTorch are sufficient to run on CPU:

```bash
pip install -r requirements.txt
python train.py
python train.py --generate --prompt "Better than BPE" --max-new-bytes 32
```

You can also use the model directly:

```python
import torch
from minblt import ByteLatentTransformer, FixedPatcher
model, patcher = ByteLatentTransformer(), FixedPatcher(4)
raw = "Hello, café!".encode("utf-8")
logits, loss = model(torch.tensor([list(raw)]), patcher)
```

Logit `t` predicts byte `x[t]` using only `x[:t]`. Inputs are bytes `0..255`, with
optional right padding `256`. A singleton BOS patch supplies the initial context.

## training

`python train.py` trains the entropy model, freezes it, then trains BLT on
`tests/taylorswift.txt`. The default is 20 steps per stage; a short CPU run is:

```bash
python train.py --steps-entropy 10 --steps-blt 10 --device cpu
```

The first 90% of the text is used for training. Validation uses the first 1,024
held-out bytes, resetting BOS per window. Loss and bits per byte count valid
bytes, not patches: `BPB = mean NLL / ln(2)`. Checkpoints go to `models/`.

Use `--stage entropy` or `--stage blt` to train separately, `--data` for another
text file, and `--patch-mode fixed --patch-size 4` for fixed patches. Dynamic
patching splits on `H[t] > threshold`; `--policy relative --threshold 0.01` instead
splits on entropy increases. Without `--threshold`, training calibrates a global
threshold for `--target-patch-size 4` and keeps it fixed for generation.

## generation

```python
from minblt import ByteLatentTransformer, visualize_patches
model, patcher = ByteLatentTransformer.load("models/blt.pt")
output = model.generate(b"Better than BPE", patcher, max_new_bytes=32)
print(repr(output))
print(visualize_patches(output, patcher(output)))
```

Generation samples one raw byte at a time, recomputing the prefix. Keep bytes for
storage; use `output.decode("utf-8", errors="replace")` for display. The defaults
are small, and ten training steps will not produce fluent text.

## tests

```bash
pytest -v .
```

Tests check causal alignment, patch masks, hashes, RoPE, gradients, byte loss,
checkpoint reload and training on the included text.

## credits and license

Repository style and training data come from [minbpe](https://github.com/karpathy/minbpe).
The algorithms come from the [BLT paper](https://arxiv.org/abs/2412.09871) and
[Meta's BLT code](https://github.com/facebookresearch/blt). This implementation uses
ordinary PyTorch attention and small models, without distributed training or KV caching.

MiniBLT's software is [MIT licensed](LICENSE). Upstream minbpe software is MIT;
Meta's code is [CC BY-NC 4.0](https://github.com/facebookresearch/blt/blob/main/LICENSE).
The included text is minbpe's February 16, 2024 copy of the
[Wikipedia Taylor Swift article](https://en.wikipedia.org/wiki/Taylor_Swift), by
its [contributors](https://en.wikipedia.org/w/index.php?title=Taylor_Swift&action=history),
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
