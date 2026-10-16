import math

import pytest
import torch

from minblt.base import byte_loss, decode, encode
from minblt.patch import EntropyPatcher, FixedPatcher, visualize_patches
from minblt.entropy import TinyEntropyModel
from minblt.model import ByteLatentTransformer


torch.set_num_threads(1)


def test_bytes():
    text = "A café, π and a snowman: ☃!"
    assert decode(encode(text)) == text
    assert bytes(list(bytes(range(256)))) == bytes(range(256))
    assert "\\xff" in repr(bytes([255]))


@pytest.mark.parametrize("size,lengths", [(1, [1]*7), (3, [3, 3, 1]), (20, [7])])
def test_fixed(size, lengths):
    patches = FixedPatcher(size)(list(range(7)))
    assert patches.lengths == lengths
    assert sum(patches.lengths) == 7
    assert patches.ids == [j for j, n in enumerate(lengths) for _ in range(n)]
    assert FixedPatcher(size)([1]).starts == [0]
    assert FixedPatcher(size)([]).lengths == []


def test_entropy_rules():
    h = [1.0, 1.2, 3.0, 2.0, 3.5]
    assert EntropyPatcher(None, 2.5).from_entropies(h).starts == [0, 2, 4]
    assert EntropyPatcher(None, 1.0, policy="relative").from_entropies(h).starts == [0, 2, 4]
    assert EntropyPatcher(None, 4, max_patch_length=2).from_entropies(h).starts == [0, 2, 4]
    assert EntropyPatcher(None, 3.0).from_entropies([3., 3.]).starts == [0]
    assert "average" in visualize_patches(b"abcdef", FixedPatcher(2)(b"abcdef"))


def test_bpb_padding():
    logits = torch.zeros(2, 3, 256)
    x = torch.tensor([[1, 2, 3], [4, 256, 256]])
    loss, bpb = byte_loss(logits, x)
    assert loss.item() == pytest.approx(math.log(256))
    assert bpb.item() == pytest.approx(8)
    logits[1, 1:] = torch.randn(2, 256)*10
    assert byte_loss(logits, x)[0].item() == pytest.approx(loss.item())


def test_entropy_alignment_and_streaming():
    torch.manual_seed(4)
    model = TinyEntropyModel(dim=16, layers=1).eval()
    a = list(b"abcdefghi")
    b = a[:4] + list(b"ZYXWV")
    h = model.entropies(a)
    torch.testing.assert_close(h[:5], model.entropies(b)[:5])
    patcher = EntropyPatcher(model, float(h[1:].median()), max_patch_length=4)
    full = patcher(a).starts
    online = []
    for t in range(len(a)):
        partial = patcher(a[:t] + [0])
        assert partial.starts == [s for s in full if s <= t]
        if t in partial.starts:
            online.append(t)
    assert online == full
    assert not any(p.requires_grad for p in model.parameters())
    achieved = patcher.calibrate(a, 3)
    assert achieved == patcher(a).average


def test_entropy_training():
    torch.manual_seed(2)
    model = TinyEntropyModel(dim=16, layers=1)
    x = torch.tensor([list(b"abababab")])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(x)[1].item()
    for _ in range(12):
        optimizer.zero_grad()
        _, loss = model(x)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial


def small_model():
    torch.manual_seed(8)
    return ByteLatentTransformer(local_dim=16, global_dim=32, global_layers=2,
                                 ngrams=(2, 3), hash_buckets=64).eval()


@pytest.mark.parametrize("patch_size", [1, 3, 100])
def test_causal_logits_and_same_patch(patch_size):
    model = small_model()
    x = torch.tensor([list(b"abcdefghij")])
    patcher = FixedPatcher(patch_size)
    original = model(x, patcher)[0]
    for t in [0, 1, 2, 3, 5, 9]:
        changed = x.clone()
        changed[:, t:] = (changed[:, t:] + 50) % 256
        torch.testing.assert_close(original[:, :t+1], model(changed, patcher)[0][:, :t+1],
                                   atol=2e-6, rtol=2e-6)


def test_dynamic_causality_with_changed_boundaries():
    model = small_model()
    torch.manual_seed(7)
    entropy = TinyEntropyModel(dim=16)
    x = torch.tensor([list(b"abcdefghijklmnop")])
    y = x.clone()
    y[:, 6:] = (y[:, 6:] + 80) % 256
    hx, hy = entropy.entropies(x[0].tolist()), entropy.entropies(y[0].tolist())
    differences = (hx[7:] - hy[7:]).abs()
    k = 7 + int(differences.argmax())
    patcher = EntropyPatcher(entropy, float((hx[k]+hy[k])/2))
    assert patcher(x[0].tolist()).starts != patcher(y[0].tolist()).starts
    torch.testing.assert_close(model(x, patcher)[0][:, :7], model(y, patcher)[0][:, :7],
                               atol=2e-6, rtol=2e-6)


def test_padding_shapes_and_gradients():
    model = small_model().train()
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [20, 21, 22, 256, 256, 256, 256]])
    logits, loss = model(x, FixedPatcher(2))
    assert logits.shape == (2, 7, 256) and torch.isfinite(loss)
    loss.backward()
    for module in [model.encoder, model.global_model, model.decoder]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    alone = model(x[1:2, :3], FixedPatcher(2))[0]
    torch.testing.assert_close(logits[1:2, :3], alone, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("patch_size", [1, 3, 100])
def test_teacher_forcing_and_placeholder(patch_size):
    model, patcher = small_model(), FixedPatcher(patch_size)
    raw = b"abcdefg"
    full = model(torch.tensor([list(raw)]), patcher)[0]
    for t in range(len(raw)):
        next_logits = model.next_logits(raw[:t], patcher)
        torch.testing.assert_close(full[0, t], next_logits, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(next_logits, model.next_logits(raw[:t], patcher, placeholder=255))


def test_global_influence_and_single_patch():
    model = small_model()
    x = torch.tensor([list(b"abcdef")])
    before = model(x, FixedPatcher(2))[0]
    with torch.no_grad():
        for p in model.global_model.parameters():
            p.add_(torch.randn_like(p)*0.3)
    after = model(x, FixedPatcher(2))[0]
    torch.testing.assert_close(before[:, :2], after[:, :2])
    assert (before[:, 2:] - after[:, 2:]).abs().max() > 1e-4
    assert model(torch.tensor([[1]]), FixedPatcher(10))[0].shape == (1, 1, 256)


def test_checkpoint_and_generation(tmp_path):
    model, patcher = small_model(), FixedPatcher(2)
    path = tmp_path / "model.pt"
    torch.save(dict(config=model.config, state_dict=model.state_dict()), path)
    checkpoint = torch.load(path, weights_only=True)
    loaded = ByteLatentTransformer(**checkpoint["config"]).eval()
    loaded.load_state_dict(checkpoint["state_dict"])
    x = torch.tensor([list(b"hello")])
    torch.testing.assert_close(model(x, patcher)[0], loaded(x, patcher)[0])
    for prompt in [b"", b"hello", "café".encode()]:
        output = loaded.generate(prompt, patcher, max_new_bytes=7, top_k=5)
        assert isinstance(output, bytes) and output.startswith(prompt)
        assert len(output) == len(prompt)+7


def test_dynamic_generation_alignment():
    model = small_model()
    entropy = TinyEntropyModel(dim=16)
    raw = b"test bytes"
    h = entropy.entropies(list(raw))
    patcher = EntropyPatcher(entropy, float(h.median()))
    full = model(torch.tensor([list(raw)]), patcher)[0]
    for t in range(len(raw)):
        torch.testing.assert_close(full[0, t], model.next_logits(raw[:t], patcher, 255),
                                   atol=2e-6, rtol=2e-6)
    assert len(model.generate(b"", patcher, 4)) == 4
