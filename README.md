# minBLT

A minimal PyTorch implementation of [Byte Latent Transformer (BLT)](https://arxiv.org/abs/2412.09871), inspired by Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe). It follows the original [BLT code](https://github.com/facebookresearch/blt) and uses the same training text as minbpe.

BLT predicts UTF-8 bytes directly, without a learned tokenizer. A local encoder turns bytes into patch representations, a global Transformer works across patches, and a local decoder predicts the next byte. Patches can have a fixed size or be chosen from next-byte entropy.

```text
bytes -> local encoder -> patch states -> global Transformer
  |                                            |
  +----------> local decoder <-----------------+
                       |
                  byte logits
```

The model includes multiple queries per patch, byte n-gram hash embeddings, RoPE, and SwiGLU. The decoder uses shifted byte states and context from completed patches so that predictions remain causal.

The main files are:

1. [minblt/base.py](minblt/base.py): Attention, normalization, and Transformer blocks.
2. [minblt/patch.py](minblt/patch.py): Fixed and entropy-based patching.
3. [minblt/entropy.py](minblt/entropy.py): A small next-byte model for estimating entropy.
4. [minblt/blt.py](minblt/blt.py): Local encoder, global Transformer, local decoder, and generation.
5. [train.py](train.py): Training and sampling on the included text.

See [lecture.md](lecture.md) and [exercise.md](exercise.md) for a closer look at the code.

## quick start

Python 3.11+ and PyTorch are enough to run the examples on CPU.

```bash
pip install -r requirements.txt
python train.py
python train.py --generate --prompt "Better than BPE" --max-new-bytes 32
```

You can also call the model directly:

```python
import torch
from minblt import ByteLatentTransformer, FixedPatcher

model = ByteLatentTransformer()
patcher = FixedPatcher(4)
raw = "Hello, café!".encode("utf-8")
logits, loss = model(torch.tensor([list(raw)]), patcher)
```

Each logit at position `t` predicts byte `t` using only preceding bytes. The model uses byte IDs `0..255`, with `256` reserved for optional padding.

## training

`python train.py` first trains an entropy model, then freezes it and trains BLT on [tests/taylorswift.txt](tests/taylorswift.txt). The defaults are deliberately small (20 steps per stage). For an even shorter CPU run:

```bash
python train.py --steps-entropy 10 --steps-blt 10 --device cpu
```

The first 90% of the text is used for training. Validation runs on the first 1,024 held-out bytes. Loss is measured per byte, with `BPB = mean NLL / ln(2)`. Checkpoints are saved to `models/`.

Use `--stage entropy` or `--stage blt` to train either model separately. Fixed patching is available with `--patch-mode fixed --patch-size 4`. By default, entropy patching chooses boundaries using a calibrated threshold for `--target-patch-size 4`.

## generation

```python
from minblt import ByteLatentTransformer, visualize_patches

model, patcher = ByteLatentTransformer.load("models/blt.pt")
output = model.generate(b"Better than BPE", patcher, max_new_bytes=32)
print(output.decode("utf-8", errors="replace"))
print(visualize_patches(output, patcher(output)))
```

Generation produces one byte at a time. The default training run is just a demo; it is not enough to generate fluent text.

## tests

```bash
pytest -v .
```

The tests cover patch boundaries, attention masks, causal predictions, gradients, and checkpoint loading.

## credits and license

The layout, coding style, and training text are inspired by [minbpe](https://github.com/karpathy/minbpe). The architecture follows the [BLT paper](https://arxiv.org/abs/2412.09871) and [Meta's reference implementation](https://github.com/facebookresearch/blt). This repository uses plain PyTorch and small models rather than Meta's distributed training setup.

This implementation is [MIT licensed](LICENSE). The original minbpe code is MIT licensed; Meta's BLT code is [CC BY-NC 4.0](https://github.com/facebookresearch/blt/blob/main/LICENSE). The included text comes from minbpe's snapshot of the [Wikipedia Taylor Swift article](https://en.wikipedia.org/wiki/Taylor_Swift), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by its [contributors](https://en.wikipedia.org/w/index.php?title=Taylor_Swift&action=history).