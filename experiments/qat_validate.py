"""Does block-wise reconstruction actually beat naive PTQ rounding?

Runs both on the same real trained blocks and compares output fidelity against
the unquantized teacher. If the trained cosine does not clearly exceed the PTQ
cosine, the whole QAT approach is not earning its complexity and we should say
so rather than shipping it.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from airllm_ternary.reconstruct import reconstruct_model

# Varied prose so the calibration activations are not degenerate.
CALIBRATION_TEXT = [
    "The transformer architecture replaced recurrence with self-attention, "
    "letting every position attend to every other position in a single step.",
    "Quantization reduces the precision of model weights. The central question "
    "is always which errors the network can absorb and which it cannot.",
    "She walked down to the harbour before dawn, when the boats were still "
    "dark shapes against the water and nothing had started moving yet.",
    "def binary_search(items, target):\n    low, high = 0, len(items) - 1\n"
    "    while low <= high:\n        mid = (low + high) // 2",
    "The mitochondrion generates most of the chemical energy needed to power "
    "the cell's biochemical reactions, stored as adenosine triphosphate.",
    "In 1687 Newton published the Principia, which set out the laws of motion "
    "and universal gravitation that dominated physics for two centuries.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--blocks", type=int, default=4, help="how many to test")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="qat_validation.json")
    args = parser.parse_args()

    print(f"loading {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).to(args.device).eval()

    batches = []
    for text in CALIBRATION_TEXT:
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=args.seq_len, padding="max_length",
        )
        batches.append({k: v.to(args.device) for k, v in encoded.items()})

    blocks = model.model.layers[: args.blocks]
    print(f"reconstructing {len(blocks)} of {len(model.model.layers)} blocks, "
          f"{args.steps} steps each, lr={args.lr}, group={args.group_size}\n")

    results = reconstruct_model(
        model, blocks, batches,
        group_size=args.group_size, steps=args.steps, lr=args.lr,
        device=args.device, compare_baseline=True, verbose=True,
    )

    print("\n" + "=" * 66)
    print(f"{'block':>5}  {'PTQ':>8}  {'trained':>8}  {'gain':>8}  {'mse drop':>9}")
    print("-" * 66)
    for row in results:
        print(f"{row['block']:>5}  {row['naive_cosine']:>8.4f}  "
              f"{row['output_cosine']:>8.4f}  {row['cosine_gain']:>+8.4f}  "
              f"{row['mse_reduction']:>8.1f}x")
    print("=" * 66)

    mean_naive = sum(r["naive_cosine"] for r in results) / len(results)
    mean_trained = sum(r["output_cosine"] for r in results) / len(results)
    print(f"\nmean PTQ cosine:     {mean_naive:.4f}")
    print(f"mean trained cosine: {mean_trained:.4f}")
    print(f"mean gain:           {mean_trained - mean_naive:+.4f}")

    verdict = (
        "QAT reconstruction is worth it"
        if mean_trained - mean_naive > 0.02
        else "NO MEANINGFUL GAIN - reconsider the approach"
    )
    print(f"\nverdict: {verdict}")

    with open(args.out, "w") as handle:
        json.dump(
            {"results": results, "mean_naive": mean_naive,
             "mean_trained": mean_trained, "config": vars(args)},
            handle, indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
