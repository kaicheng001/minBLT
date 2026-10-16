import math
import hashlib
from pathlib import Path

import pytest
import torch

from minblt.base import byte_loss, decode, encode
from minblt.patch import EntropyPatcher, FixedPatcher, Patches, visualize_patches
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
    logits = torch.randn(2, 3, 256)
    expected = -logits.log_softmax(-1)[x != 256].gather(1, x[x != 256, None]).sum()
    loss, bpb = byte_loss(logits, x)
    assert loss.item() == pytest.approx(expected.item()/4)
    assert bpb.item() == pytest.approx(expected.item()/(4*math.log(2)))


@pytest.mark.parametrize("policy", ["global", "relative"])
def test_entropy_alignment_and_streaming(policy):
    torch.manual_seed(4)
    model = TinyEntropyModel(dim=16, layers=1).eval()
    a = list(b"abcdefghi")
    b = a[:4] + list(b"ZYXWV")
    h = model.entropies(a)
    torch.testing.assert_close(h[:5], model.entropies(b)[:5])
    scores = h[1:] if policy == "global" else h[1:]-h[:-1]
    patcher = EntropyPatcher(model, float(scores.median()), policy=policy, max_patch_length=4)
    full = patcher(a).starts
    online = []
    for t in range(len(a)):
        partial = patcher(a[:t] + [0])
        assert partial.starts == [s for s in full if s <= t]
        if t in partial.starts:
            online.append(t)
    assert online == full
    assert not any(p.requires_grad for p in model.parameters())
    if policy == "global":
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
        gradients = [p.grad for p in module.parameters()]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
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


def test_entropy_checkpoint_roundtrip(tmp_path):
    from demo import load_checkpoint
    from train import pack

    model = small_model()
    entropy = TinyEntropyModel(dim=16)
    patcher = EntropyPatcher(entropy, 5.35, max_patch_length=4)
    checkpoint = pack(model)
    checkpoint.update(entropy=pack(entropy), patcher=dict(
        mode="entropy", threshold=5.35, policy="global", max_patch_length=4))
    path = tmp_path / "blt.pt"
    torch.save(checkpoint, path)
    restored, restored_patcher = load_checkpoint(path)
    raw = b"bytes and patches"
    assert patcher(raw) == restored_patcher(raw)
    assert not any(p.requires_grad for p in restored_patcher.model.parameters())
    x = torch.tensor([list(raw)])
    torch.testing.assert_close(model(x, patcher)[0], restored(x, restored_patcher)[0])
    torch.manual_seed(123)
    before = model.generate(raw, patcher, 4)
    torch.manual_seed(123)
    assert before == restored.generate(raw, restored_patcher, 4)


def test_hash_and_patch_pool_membership():
    model = small_model()
    x = torch.tensor([list(b"abcdefgh")])
    changed = x.clone()
    changed[:, 4:] += 10
    torch.testing.assert_close(model.encoder.hash(x)[:, :4], model.encoder.hash(changed)[:, :4])
    patches = FixedPatcher(4)(x[0].tolist())
    _, original = model.encoder(x, patches)
    _, mutated = model.encoder(changed, patches)
    torch.testing.assert_close(original[:, :1], mutated[:, :1])
    assert (original[:, 1:] - mutated[:, 1:]).abs().max() > 0
    # No hash tables is a supported, deterministic ablation.
    assert ByteLatentTransformer(ngrams=())(x, FixedPatcher(2))[0].shape == (1, 8, 256)


@pytest.mark.parametrize("starts,size", [([1], 3), ([0, 0], 3), ([0, 4], 3), ([0, 2, 1], 3)])
def test_invalid_patch_metadata(starts, size):
    with pytest.raises(ValueError):
        Patches.build(starts, size)


@pytest.mark.parametrize("x", [[[256]], [[1, 256, 2]], [[-1]], [[257]]])
def test_invalid_byte_batches(x):
    with pytest.raises(ValueError):
        small_model()(torch.tensor(x), FixedPatcher(2))


def test_original_minbpe_dataset():
    from train import DEFAULT_DATA
    raw = (Path(__file__).parent / "taylorswift.txt").read_bytes()
    assert len(raw) == 185768
    assert hashlib.sha256(raw).hexdigest() == "c2e39cb822d4ae0caac22152cefc306d466e31217a9c5524e493ad2b76792f57"
    assert DEFAULT_DATA == "tests/taylorswift.txt"


def test_rope_scalar_reference():
    from minblt import base
    x = torch.tensor([[[[1., 2., 3., 4.], [2., 1., 4., 3.]]]])
    actual = base.rope(x)
    expected = x.clone()
    for t in range(2):
        for pair in range(2):
            angle = t * 10000**(-2*pair/4)
            a, b = x[0, 0, t, 2*pair:2*pair+2].tolist()
            expected[0, 0, t, 2*pair] = a*math.cos(angle)-b*math.sin(angle)
            expected[0, 0, t, 2*pair+1] = a*math.sin(angle)+b*math.cos(angle)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))


def test_distribution_entropy():
    from minblt.entropy import entropy
    assert entropy(torch.zeros(3, 256)).tolist() == pytest.approx([math.log(256)]*3)
    logits = torch.full((256,), -100.)
    logits[10] = 100
    assert entropy(logits).item() < 1e-6
    logits = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64)
    weights = [math.exp(v) for v in logits.tolist()]
    probabilities = [w/sum(weights) for w in weights]
    assert entropy(logits).item() == pytest.approx(-sum(p*math.log(p) for p in probabilities))


def test_reference_initial_patch_and_decoder_ids():
    from minblt.patch import decoder_patch_ids
    raw = Patches.build([0, 3, 5], 7)
    internal = raw.with_bos()
    assert internal.starts == [0, 1, 4, 6]
    assert internal.lengths == [1, 3, 2, 2]
    assert internal.ids == [0, 1, 1, 1, 2, 2, 3, 3]
    assert decoder_patch_ids(internal) == [0, 0, 0, 1, 1, 2, 2]
    # Meta find_entropy_patch_start_ids uses initial encoder starts [0,1].
    # Its monotonic helper additionally forces its first surviving candidate.
    h = [1., 1.2, 3., 2., 3.5]
    assert EntropyPatcher(None, 1., policy="relative").from_entropies(h).starts == [0, 1, 2, 4]
    assert EntropyPatcher(None, 99., policy="relative").from_entropies(h).starts == [0, 1]
    assert EntropyPatcher(None, 99.).from_entropies(h).with_bos().lengths == [1, 5]


def test_patch_masks_independent():
    from minblt.patch import patch_mask
    q = torch.tensor([0, 0, 1, 1, 2, 2])
    k = torch.tensor([0, 1, 1, 2])
    expected = [[a == b for b in k.tolist()] for a in q.tolist()]
    assert patch_mask(q, k)[0].tolist() == expected
    expected = [[0 <= a-b < 2 for b in k.tolist()] for a in q.tolist()]
    assert patch_mask(q, k, window=2)[0].tolist() == expected


@pytest.mark.parametrize("n", [1, 3, 5, 8])
@pytest.mark.parametrize("function", [0, 1])
def test_hash_scalar_reference(n, function):
    from minblt.blt import byte_hash
    # Internal symbols use Meta's offset=4 and BOS=1; zero left padding is BOE.
    values = [1, 4, 259, 5, 42, 190, 255, 128]
    prime = [1000000007, 5915587277][function]
    expected = []
    for t in range(len(values)):
        gram = ([0]*(n-1) + values[:t+1])[-n:]
        integer = sum(v*prime**i for i, v in enumerate(gram)) % (1 << 64)
        if integer >= 1 << 63:
            integer -= 1 << 64
        expected.append(integer % 509)
    assert byte_hash(torch.tensor([values]), n, 509, function).tolist() == [expected]


@pytest.mark.parametrize("queries", [1, 2, 3])
def test_multiple_patch_queries_and_layers(queries):
    model = ByteLatentTransformer(local_dim=16, global_dim=32, encoder_layers=2,
                                 decoder_layers=2, global_layers=3, cross_attn_k=queries,
                                 ngrams=(3, 4), hash_buckets=64)
    raw = b"abcdefgh"
    patches = FixedPatcher(3)(raw).with_bos()
    internal = torch.tensor([[1]+[b+4 for b in raw]])
    h, p = model.encoder(internal, patches)
    assert h.shape == (1, 9, 16)
    assert p.shape == (1, 4, queries, 16)
    assert len(model.encoder.blocks) == len(model.decoder.blocks) == 2
    x = torch.tensor([list(raw)])
    _, loss = model(x, FixedPatcher(3))
    loss.backward()
    for module in [model.encoder, model.global_model, model.decoder]:
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
