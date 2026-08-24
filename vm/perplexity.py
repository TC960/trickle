"""Wikitext perplexity harness — the number that settles whether a method works.

Every quality claim in this project so far has been per-block output cosine,
which cannot see error compounding across depth. Perplexity can. This is the
common yardstick every compression method gets measured against.

    python perplexity.py --model google/gemma-4-31B
    python perplexity.py --model X --quant int8
    python perplexity.py --model X --quant nf4 --tag qlora-base
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer



def peak_mem_gb() -> float:
    """Peak device memory in GB, or process RSS when no accelerator is present."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    import resource, sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(raw / 1e9 if sys.platform == "darwin" else raw / 1e6, 3)


def load_wikitext(tokenizer, split="test", limit_chars=None, config="wikitext-2-raw-v1"):
    """Concatenate wikitext into one long token sequence, as is standard.

    The bare `wikitext` dataset id no longer resolves -- newer huggingface_hub
    requires a namespaced repo -- so try the canonical mirror first and fall
    back through known-equivalent copies.
    """
    from datasets import load_dataset

    candidates = ["Salesforce/wikitext", "EleutherAI/wikitext_document_level"]
    last_error = None
    for repo in candidates:
        try:
            data = load_dataset(repo, config, split=split)
            print(f"  dataset: {repo}/{config}", flush=True)
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError(f"could not load wikitext from {candidates}: {last_error}")

    column = "text" if "text" in data.column_names else data.column_names[0]
    text = "\n\n".join(data[column])
    if limit_chars:
        text = text[:limit_chars]
    ids = tokenizer(text, return_tensors="pt").input_ids
    return ids, len(text.encode("utf-8"))


@torch.inference_mode()
def _nll_sum(model, chunk, targets, vocab_chunk=256):
    """Summed NLL over the unmasked targets, without a full-vocab float32 CE.

    Passing `labels=` to the model makes transformers flatten the logits and
    call cross_entropy on the whole window at once. For Gemma 4 that is
    2048 x 262144 upcast to float32 = 2.1 GB for one tensor, on top of the
    bf16 logits -- which is exactly what OOM'd the published-checkpoint eval on
    an 80 GB card. Slicing the sequence keeps the upcast bounded while
    computing the identical quantity.

    Returns (summed_nll, n_valid_targets).
    """
    logits = model(chunk).logits
    shift_logits = logits[:, :-1]
    shift_targets = targets[:, 1:]
    vocab = shift_logits.shape[-1]

    total, valid = 0.0, 0
    for s in range(0, shift_logits.shape[1], vocab_chunk):
        lg = shift_logits[:, s:s + vocab_chunk].reshape(-1, vocab)
        tg = shift_targets[:, s:s + vocab_chunk].reshape(-1)
        mask = tg != -100
        n = int(mask.sum())
        if n == 0:
            continue
        total += F.cross_entropy(lg[mask].float(), tg[mask],
                                 reduction="sum").item()
        valid += n
    return total, valid


@torch.inference_mode()
def perplexity(model, input_ids, window=2048, stride=None, device=None,
               verbose=True, n_bytes=None, check_loss=True):
    """Sliding-window perplexity.

    Only the newly-revealed tokens in each window contribute to the loss, so
    every token is scored with the maximum context available to it. Using
    stride == window (no overlap) is cheaper but inflates perplexity, so the
    default overlaps by half.
    """
    stride = stride or window // 2
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    started = time.time()

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + window, input_ids.size(1))
        target_len = end - previous_end
        if target_len <= 0:
            continue

        chunk = input_ids[:, begin:end].to(device or model.device)
        targets = chunk.clone()
        # Mask everything the previous window already scored.
        targets[:, :-target_len] = -100

        if (targets[:, 1:] != -100).sum().item() == 0:
            continue
        nll, valid = _nll_sum(model, chunk, targets)
        if valid == 0:
            continue

        # Verify once per call that the memory-lean path agrees with the
        # reference `labels=` path, so switching it can't silently move every
        # number in this project. Skipped if the reference itself OOMs.
        if check_loss and total_tokens == 0:
            try:
                ref = model(chunk, labels=targets).loss.float().item() * valid
                drift = abs(ref - nll) / max(abs(ref), 1e-9)
                print(f"    loss-path check: chunked {nll/valid:.6f} vs "
                      f"labels= {ref/valid:.6f}  (rel {drift:.2e})", flush=True)
                if drift > 1e-4:
                    raise RuntimeError(
                        f"chunked NLL disagrees with labels= path by {drift:.2e}")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("    loss-path check skipped (reference path OOM)", flush=True)

        total_nll += nll
        total_tokens += valid
        previous_end = end

        if verbose and begin % (stride * 20) == 0:
            running = torch.exp(torch.tensor(total_nll / max(total_tokens, 1)))
            print(f"    {end:>7}/{input_ids.size(1)} tok  ppl~{running:.4f}", flush=True)

        if end == input_ids.size(1):
            break

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    # Bits-per-byte normalizes by raw UTF-8 bytes rather than tokens, so it IS
    # comparable across models with different tokenizers -- unlike perplexity,
    # which measures per-token surprise and therefore depends on how the
    # tokenizer chops the text. Required for any vocab-size comparison.
    bpb = (total_nll / n_bytes / 0.6931471805599453) if n_bytes else None

    return {
        "perplexity": ppl,
        "bits_per_byte": round(bpb, 5) if bpb else None,
        "eval_bytes": n_bytes,
        "tokens_scored": total_tokens,
        "window": window,
        "stride": stride,
        "eval_seconds": round(time.time() - started, 1),
    }


def build_model(model_id, quant, dtype=torch.bfloat16):
    """Load a model, optionally with an on-the-fly quantization config."""
    kwargs = dict(dtype=dtype, device_map="auto", low_cpu_mem_usage=True)

    if quant in ("int8", "nf4", "fp4"):
        from transformers import BitsAndBytesConfig

        if quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4" if quant == "nf4" else "fp4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        kwargs.pop("dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", default="none",
                        choices=["none", "int8", "nf4", "fp4"])
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--limit-chars", type=int, default=None,
                        help="truncate the eval text; use for quick smoke runs")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default="results.jsonl")
    args = parser.parse_args()

    tag = args.tag or f"{args.model.split('/')[-1]}-{args.quant}"
    print(f"=== {tag} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    started = time.time()
    model = build_model(args.model, args.quant)
    load_seconds = time.time() - started

    footprint = None
    try:
        footprint = model.get_memory_footprint() / 1e9
    except Exception:
        pass
    print(f"  loaded in {load_seconds:.0f}s, footprint "
          f"{footprint:.2f} GB" if footprint else f"  loaded in {load_seconds:.0f}s",
          flush=True)

    input_ids = load_wikitext(tokenizer, limit_chars=args.limit_chars)
    print(f"  eval tokens: {input_ids.size(1)}", flush=True)

    metrics = perplexity(model, input_ids, args.window, args.stride,
                         n_bytes=n_bytes)
    record = {
        "tag": tag,
        "model": args.model,
        "quant": args.quant,
        "load_seconds": round(load_seconds, 1),
        "footprint_gb": round(footprint, 3) if footprint else None,
        "peak_gpu_gb": peak_mem_gb(),
        **metrics,
    }

    print(f"\n  PERPLEXITY {record['perplexity']:.4f}  "
          f"BPB {record.get('bits_per_byte')}  "
          f"footprint {record['footprint_gb']} GB   "
          f"peak {record['peak_gpu_gb']} GB\n", flush=True)

    with open(args.out, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"  appended to {args.out}")


if __name__ == "__main__":
    main()
