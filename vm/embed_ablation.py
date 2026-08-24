"""Embedding-table compression ablation.

This project's measured bottleneck is not the transformer layers -- it is the
embedding table. On BitNet-2B the table is 657 MB while all 30 quantized layers
together are 522 MB, so peak memory floors out regardless of how aggressively
the layers are streamed.

The complication is weight tying: `lm_head` usually shares the embedding matrix,
so compressing it damages both token lookup AND every output logit. Those two
uses have very different sensitivity, which suggests an obvious experiment
nobody seems to have published cleanly:

    tied     compress the shared matrix, both uses degrade
    untied   compress the input lookup only, keep a full-precision output head

Untying costs memory back, so the question is whether the quality recovered is
worth it. Each method is measured by wikitext perplexity, not by weight error.

    python embed_ablation.py --model X --method int8
    python embed_ablation.py --model X --method int4 --untie
    python embed_ablation.py --model X --method svd --rank 1024
"""

import argparse
import copy
import json
import time

import torch
import torch.nn as nn

from perplexity import load_wikitext, perplexity



def peak_mem_gb() -> float:
    """Peak device memory in GB, or process RSS when no accelerator is present."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    import resource, sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(raw / 1e9 if sys.platform == "darwin" else raw / 1e6, 3)


def quantize_rowwise(weight, bits, group_size=None):
    """Symmetric per-row (per-token) quantization, optionally grouped.

    Per-row is the natural choice for an embedding table: each row is one
    token's vector, rows have wildly different norms, and a shared scale would
    be dominated by whichever tokens happen to have large embeddings.
    """
    original_dtype = weight.dtype
    weight = weight.float()
    qmax = 2 ** (bits - 1) - 1

    if group_size:
        rows, cols = weight.shape
        pad = (-cols) % group_size
        if pad:
            weight = torch.nn.functional.pad(weight, (0, pad))
        grouped = weight.reshape(rows, -1, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        dequant = (grouped / scale).round().clamp(-qmax - 1, qmax) * scale
        weight = dequant.reshape(rows, -1)[:, :cols]
        n_scales = scale.numel()
    else:
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        weight = (weight / scale).round().clamp(-qmax - 1, qmax) * scale
        n_scales = scale.numel()

    return weight.to(original_dtype), n_scales


def svd_compress(weight, rank):
    """Low-rank factorization: [V, H] -> [V, r] @ [r, H].

    Only shrinks anything when r < V*H/(V+H). Embeddings are plausible
    candidates because token vectors are known to occupy a low-dimensional
    subspace, but the reconstruction is global rather than per-token, so rare
    tokens tend to suffer most.
    """
    original_dtype = weight.dtype
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    approx = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    stored = rank * (weight.shape[0] + weight.shape[1])
    return approx.to(original_dtype), stored


def apply_method(model, method, bits, group_size, rank, untie):
    """Compress the embedding table in place. Returns a stats dict."""
    embed = model.get_input_embeddings()
    # clone(): copy_ below overwrites this storage, and we need the true
    # pre-compression values to measure reconstruction error against.
    original = embed.weight.data.clone()
    vocab, hidden = original.shape
    original_bytes = original.numel() * original.element_size()

    head = model.get_output_embeddings()
    # Compare against the LIVE embedding tensor, not `original` -- that is a
    # clone, so its data_ptr never matches the head and was_tied was always
    # False, silently disabling --untie.
    was_tied = (head is not None
                and head.weight.data_ptr() == embed.weight.data_ptr())

    # Preserve a full-precision output head before touching the shared matrix.
    if untie and was_tied and head is not None:
        head.weight = nn.Parameter(original.clone(), requires_grad=False)
        model.config.tie_word_embeddings = False

    if method == "none":
        compressed, stored_elements = original, original.numel()
    elif method in ("int8", "int4"):
        bits = 8 if method == "int8" else 4
        compressed, n_scales = quantize_rowwise(original, bits, group_size)
        stored_elements = original.numel() * bits / 16 + n_scales
    elif method == "svd":
        compressed, stored_elements = svd_compress(original, rank)
    else:
        raise ValueError(method)

    embed.weight.data.copy_(compressed)

    # Re-tie so the (now compressed) matrix is shared again, unless untying.
    if was_tied and not untie and head is not None:
        head.weight = embed.weight

    compressed_bytes = stored_elements * 2  # bf16-equivalent accounting
    extra = original_bytes if (untie and was_tied) else 0

    return {
        "vocab": vocab,
        "hidden": hidden,
        "was_tied": was_tied,
        "untied": bool(untie and was_tied),
        "embed_mb_before": round(original_bytes / 1e6, 1),
        "embed_mb_after": round(compressed_bytes / 1e6, 1),
        "untied_head_mb": round(extra / 1e6, 1),
        "net_mb": round((compressed_bytes + extra) / 1e6, 1),
        "reconstruction_cosine": torch.nn.functional.cosine_similarity(
            original.float().flatten()[:10_000_000],
            compressed.float().flatten()[:10_000_000], dim=0
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="int8",
                        choices=["none", "int8", "int4", "svd"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--untie", action="store_true")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--limit-chars", type=int, default=None)
    parser.add_argument("--out", default="embed_results.jsonl")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    label = (f"{args.method}"
             + (f"-g{args.group_size}" if args.group_size else "")
             + (f"-r{args.rank}" if args.method == "svd" else "")
             + ("-untied" if args.untie else ""))
    print(f"=== embed:{label} on {args.model} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    stats = apply_method(model, args.method, args.bits,
                         args.group_size, args.rank, args.untie)
    print("  " + json.dumps(stats), flush=True)

    input_ids, n_bytes = load_wikitext(tokenizer, limit_chars=args.limit_chars)
    started = time.time()
    metrics = perplexity(model, input_ids, args.window, n_bytes=n_bytes)

    record = {
        "tag": f"{args.model.split(chr(47))[-1]}-embed-{label}",
        "model": args.model,
        "method": args.method,
        "group_size": args.group_size,
        "rank": args.rank if args.method == "svd" else None,
        **stats,
        **metrics,
        "peak_gpu_gb": peak_mem_gb(),
    }
    print(f"\n  PERPLEXITY {record['perplexity']:.4f}  "
          f"embed {stats['embed_mb_before']} -> {stats['net_mb']} MB\n", flush=True)

    with open(args.out, "a") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
