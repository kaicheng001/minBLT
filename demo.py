"""Reload the tiny trained model and sample raw bytes without network access."""

import argparse
import shlex

import torch

from minblt import ByteLatentTransformer, EntropyPatcher, FixedPatcher, TinyEntropyModel, visualize_patches


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = ByteLatentTransformer(**saved["config"]).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    config = saved["patcher"].copy()
    mode = config.pop("mode")
    if mode == "fixed":
        patcher = FixedPatcher(**config)
    else:
        payload = saved["entropy"]
        entropy = TinyEntropyModel(**payload["config"]).to(device)
        entropy.load_state_dict(payload["state_dict"])
        patcher = EntropyPatcher(entropy, **config)
    return model, patcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/blt.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="Better than BPE")
    parser.add_argument("--max-new-bytes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model, patcher = load_checkpoint(args.checkpoint, args.device)
    prompt = args.prompt.encode("utf-8")
    output = model.generate(prompt, patcher, args.max_new_bytes, args.temperature, args.top_k)
    print("prompt: " + ascii(args.prompt))
    print("generated (decoded, escaped): " + ascii(output[len(prompt):].decode("utf-8", errors="replace")))
    print("generated (exact raw bytes): " + repr(output[len(prompt):]))
    print(f"prompt bytes={len(prompt)}, new bytes={len(output)-len(prompt)}, total bytes={len(output)}")
    print(visualize_patches(output, patcher(output)))
    command = ["python", "demo.py", "--checkpoint", args.checkpoint, "--device", args.device,
               "--prompt", args.prompt, "--max-new-bytes", str(args.max_new_bytes),
               "--seed", str(args.seed), "--temperature", str(args.temperature), "--top-k", str(args.top_k)]
    print("Reproduce (POSIX shell): " + shlex.join(command))


if __name__ == "__main__":
    main()
