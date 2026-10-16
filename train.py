"""Two small training stages: a byte entropy LM, then a frozen-patcher BLT."""

import argparse
import math
from pathlib import Path

import torch

from minblt import ByteLatentTransformer, EntropyPatcher, FixedPatcher, TinyEntropyModel


def pack(model):
    return dict(config=model.config, state_dict=model.state_dict())


def fit(model, data, validation, steps, args, patcher=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(steps):
        starts = torch.randint(len(data)-args.seq_len+1, (args.batch_size,))
        batch = torch.stack([data[int(s):int(s)+args.seq_len] for s in starts])
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch) if patcher is None else model(batch, patcher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step+1) % max(1, steps//5) == 0 or step+1 == steps:
            stats = ""
            if patcher is not None:
                patches = [patcher(row.tolist()) for row in batch]
                count = sum(len(p.starts) for p in patches)
                stats = f", patches={count}, average={batch.numel()/count:.2f} bytes"
            print(f"step {step+1}/{steps}: loss={loss.item():.4f}{stats}")
    model.eval()
    # Evaluate every held-out byte once, with a BOS reset per small window.
    total_nll, count = 0.0, 0
    with torch.no_grad():
        for chunk in validation.split(args.seq_len):
            x = chunk[None]
            _, loss = model(x) if patcher is None else model(x, patcher)
            total_nll += loss.item()*chunk.numel()
            count += chunk.numel()
    print(f"held-out BPB={total_nll/(count*math.log(2)):.4f} ({count} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["entropy", "blt", "all"], default="all")
    parser.add_argument("--data", default="tests/input.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps-entropy", type=int, default=100)
    parser.add_argument("--steps-blt", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-mode", choices=["entropy", "fixed"], default="entropy")
    parser.add_argument("--policy", choices=["global", "relative"], default="global")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--target-patch-size", type=float, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--max-patch-length", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps_entropy, args.steps_blt) < 0 or min(args.seq_len, args.batch_size) < 1:
        parser.error("Step counts must be nonnegative; sequence and batch sizes must be positive")
    if args.policy == "relative" and args.threshold is None and args.patch_mode == "entropy":
        parser.error("Relative patching requires an explicit --threshold in nats")
    torch.set_num_threads(1)  # Tiny CPU matrices are faster without thread oversubscription.
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    raw = Path(args.data).read_bytes()
    split = int(0.9*len(raw))
    if split < args.seq_len or split == len(raw):
        parser.error("Data must provide a full training window and a held-out split")
    data = torch.tensor(list(raw[:split]), dtype=torch.long, device=device)
    validation = torch.tensor(list(raw[split:]), dtype=torch.long, device=device)
    directory = Path("models")
    directory.mkdir(exist_ok=True)
    print("Tiny-model educational measurements; no paper-reproduction claim.")
    entropy = None
    if args.stage in ("entropy", "all"):
        print("Training the separate entropy model")
        entropy = TinyEntropyModel().to(device)
        fit(entropy, data, validation, args.steps_entropy, args)
        torch.save(pack(entropy), directory / "entropy.pt")
        print("Saved models/entropy.pt")
    if args.stage in ("blt", "all"):
        if args.patch_mode == "entropy":
            if entropy is None:
                path = directory / "entropy.pt"
                if not path.exists():
                    parser.error("Train the entropy stage first, or use --stage all")
                saved = torch.load(path, map_location=device, weights_only=True)
                entropy = TinyEntropyModel(**saved["config"]).to(device)
                entropy.load_state_dict(saved["state_dict"])
            patcher = EntropyPatcher(entropy, args.threshold if args.threshold is not None else 0,
                                     args.policy, args.max_patch_length)
            # Calibrate only on training bytes, in windows with the same BOS reset
            # semantics as training. Never re-fit on validation or generated text.
            calibration = data[:min(len(data), args.seq_len)].tolist()
            if args.threshold is None:
                achieved = patcher.calibrate(calibration, args.target_patch_size)
                print(f"Calibration target={args.target_patch_size:.2f}, achieved={achieved:.2f} bytes")
            print(f"Frozen entropy patcher: policy={args.policy}, threshold={patcher.threshold:.6f} nats")
            patch_config = dict(mode="entropy", threshold=patcher.threshold, policy=args.policy,
                                max_patch_length=args.max_patch_length)
        else:
            patcher = FixedPatcher(args.patch_size)
            patch_config = dict(mode="fixed", patch_size=args.patch_size)
        model = ByteLatentTransformer().to(device)
        print("Training BLT with " + args.patch_mode + " patches")
        fit(model, data, validation, args.steps_blt, args, patcher)
        checkpoint = pack(model)
        checkpoint.update(patcher=patch_config, entropy=pack(entropy) if entropy is not None else None)
        torch.save(checkpoint, directory / "blt.pt")
        print("Saved models/blt.pt (self-contained model and patcher checkpoint)")


if __name__ == "__main__":
    main()
