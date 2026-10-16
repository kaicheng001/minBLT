"""Train an entropy model and BLT on minbpe's original example corpus."""

import argparse
import hashlib
import math
from pathlib import Path
import random
import time

import torch

from minblt import ByteLatentTransformer, EntropyModel, EntropyPatcher, FixedPatcher, visualize_patches

DEFAULT_DATA = "tests/taylorswift.txt"


def fit(model, data, validation, steps, args, patcher=None):
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
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
                count = sum(len(patcher(row.tolist()).starts) for row in batch)
                stats = f", patches={count}, average={batch.numel()/count:.2f} bytes"
            print(f"step {step+1}/{steps}: loss={loss.item():.4f}, BPB={loss.item()/math.log(2):.4f}{stats}")
    model.eval()
    total_nll, count = 0.0, 0
    with torch.no_grad():
        for chunk in validation.split(args.seq_len):
            _, loss = model(chunk[None]) if patcher is None else model(chunk[None], patcher)
            total_nll += loss.item()*chunk.numel()
            count += chunk.numel()
    print(f"held-out BPB={total_nll/(count*math.log(2)):.4f} ({count} bytes, BOS reset per window)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["entropy", "blt", "all"], default="all")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps-entropy", type=int, default=20)
    parser.add_argument("--steps-blt", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-bytes", type=int, default=1024)
    parser.add_argument("--patch-mode", choices=["entropy", "fixed"], default="entropy")
    parser.add_argument("--policy", choices=["global", "relative"], default="global")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--target-patch-size", type=float, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--max-patch-length", type=int, default=16)
    parser.add_argument("--checkpoint", default="models/blt.pt")
    parser.add_argument("--generate", action="store_true", help="Reload a checkpoint and sample bytes")
    parser.add_argument("--prompt", default="Better than BPE")
    parser.add_argument("--max-new-bytes", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.generate:
        model, patcher, _ = ByteLatentTransformer.load(args.checkpoint, device)
        prompt = args.prompt.encode("utf-8")
        output = model.generate(prompt, patcher, args.max_new_bytes, top_k=40)
        print("prompt: " + ascii(args.prompt))
        print("generated raw bytes: " + repr(output[len(prompt):]))
        print("generated text (display only): " + ascii(output[len(prompt):].decode("utf-8", errors="replace")))
        print(f"output length={len(output)}, new bytes={len(output)-len(prompt)}")
        print(visualize_patches(output, patcher(output)))
        return
    if min(args.steps_entropy, args.steps_blt) < 0 or min(args.seq_len, args.batch_size, args.eval_bytes) < 1:
        parser.error("Step counts must be nonnegative; sequence, batch and evaluation sizes must be positive")
    if args.policy == "relative" and args.threshold is None and args.patch_mode == "entropy":
        parser.error("Relative patching requires an explicit --threshold in nats")
    raw = Path(args.data).read_bytes()
    split = int(0.9*len(raw))
    if split < args.seq_len or split == len(raw):
        parser.error("Data must provide a full training window and a held-out split")
    metadata = dict(data=args.data, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    train_bytes=split, validation_bytes=len(raw)-split,
                    evaluated_bytes=min(args.eval_bytes, len(raw)-split), seed=args.seed,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    steps_entropy=args.steps_entropy, steps_blt=args.steps_blt)
    data = torch.tensor(list(raw[:split]), dtype=torch.long, device=device)
    validation = torch.tensor(list(raw[split:split+args.eval_bytes]), dtype=torch.long, device=device)
    Path("models").mkdir(exist_ok=True)
    start = time.monotonic()
    print(f"Corpus: {args.data}, {len(raw)} bytes; chronological 90/10 split")
    print("Small-model measurements; no paper-scale results are implied.")
    entropy_model = None
    entropy_metadata = None
    if args.stage in ("entropy", "all"):
        print("Training the separate entropy model")
        entropy_model = EntropyModel().to(device)
        fit(entropy_model, data, validation, args.steps_entropy, args)
        entropy_metadata = metadata.copy()
        torch.save(dict(config=entropy_model.config, state_dict=entropy_model.state_dict(),
                        metadata=metadata), "models/entropy.pt")
    if args.stage in ("blt", "all"):
        if args.patch_mode == "entropy":
            if entropy_model is None:
                if not Path("models/entropy.pt").exists():
                    parser.error("Train the entropy stage first, or use --stage all")
                saved = torch.load("models/entropy.pt", map_location=device, weights_only=True)
                if saved["metadata"]["sha256"] != metadata["sha256"]:
                    parser.error("Entropy checkpoint was trained on a different corpus")
                entropy_model = EntropyModel(**saved["config"]).to(device)
                entropy_model.load_state_dict(saved["state_dict"])
                entropy_metadata = saved["metadata"]
            patcher = EntropyPatcher(entropy_model, args.threshold if args.threshold is not None else 0,
                                     args.policy, args.max_patch_length)
            if args.threshold is None:
                achieved = patcher.calibrate(data[:args.seq_len].tolist(), args.target_patch_size)
                print(f"Calibration target={args.target_patch_size:.2f}, achieved={achieved:.2f} bytes/patch")
            print(f"Frozen entropy patcher: {args.policy}, threshold={patcher.threshold:.6f} nats")
        else:
            patcher = FixedPatcher(args.patch_size)
        print("Training BLT with " + args.patch_mode + " patches")
        model = ByteLatentTransformer().to(device)
        fit(model, data, validation, args.steps_blt, args, patcher)
        if entropy_metadata is not None:
            metadata["entropy_training"] = entropy_metadata
        model.save(args.checkpoint, patcher, metadata)
        print("Saved " + args.checkpoint)
    print(f"Training took {time.monotonic()-start:.2f} seconds")


if __name__ == "__main__":
    main()
