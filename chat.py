"""Interactive chat against the ternary streaming engine.

    python chat.py                      # default 0.75 GB budget
    python chat.py --budget-gb 0.05     # minimum footprint, heavy streaming
    python chat.py --budget-gb 2.0      # everything resident, fastest

Commands inside the session:
    /stats    engine counters (bytes read, evictions, cache hits)
    /reset    clear conversation history
    /raw      toggle chat-template vs raw completion
    /quit     exit
"""

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, TextStreamer

from airllm_ternary.model import load_streaming_model
from airllm_ternary.shard import load_manifest

DEFAULT_MODEL = "microsoft/bitnet-b1.58-2B-4T"

# Resolved from the environment so the repo carries no machine-specific paths.
# activate.sh exports AIRLLM_CACHE; the fallback keeps this runnable without it.
_CACHE = Path(os.environ.get("AIRLLM_CACHE", Path.home() / ".cache/airllm-ternary"))
DEFAULT_SHARDS = str(_CACHE / f"shards-{DEFAULT_MODEL.split('/')[-1]}")


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def print_stats(manager, manifest):
    report = manager.report()
    pinned_mb = manifest["shards"]["globals"]["nbytes"] / 1e6
    print(f"""
  budget          {report['budget_mb']:>8.0f} MB
  streamed layers {report['resident_mb']:>8.0f} MB resident
  pinned          {pinned_mb:>8.0f} MB  (embeddings + norms, always in memory)
  peak process    {peak_rss_gb():>8.2f} GB
  total read      {report['gb_read']:>8.2f} GB
  layer loads     {report['misses']:>8d}  ({report['evictions']} evictions)
  cache hit rate  {report['hit_rate']:>8.3f}
  prefetch hits   {report['prefetch_hits']:>8d}
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shards", default=DEFAULT_SHARDS)
    parser.add_argument("--budget-gb", type=float, default=0.75)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="raw completion instead of chat formatting")
    args = parser.parse_args()

    manifest = load_manifest(args.shards)
    total_gb = manifest["summary"]["total_bytes"] / 1e9
    largest_mb = manifest["summary"]["largest_layer_bytes"] / 1e6
    layers = manifest["summary"]["num_layer_shards"]

    print(f"model      {args.model}")
    print(f"shards     {total_gb:.2f} GB across {layers} layers "
          f"(largest {largest_mb:.0f} MB)")
    print(f"budget     {args.budget_gb:.2f} GB -> "
          f"{'STREAMING' if args.budget_gb < total_gb else 'fully resident'}")
    print(f"device     {args.device}")
    print("\nloading...", end="", flush=True)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, manager = load_streaming_model(
        args.model, args.shards,
        budget_gb=args.budget_gb, device=args.device,
        prefetch=not args.no_prefetch,
    )
    print(f" ready in {time.time() - started:.1f}s")

    use_chat = tokenizer.chat_template is not None and not args.raw
    print(f"mode       {'chat' if use_chat else 'raw completion'}")

    # Belt and braces: the generation config should already carry these, but an
    # unset stop token makes the model role-play both sides of the conversation,
    # so pin them explicitly here too.
    stop_ids = set()
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is not None:
        stop_ids.update(configured if isinstance(configured, list) else [configured])
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            stop_ids.add(token_id)
    stop_ids.discard(None)
    print(f"stop tokens {sorted(stop_ids)}")

    # The checkpoint's generation_config sets max_length=4096, which makes
    # transformers warn on every call once we also pass max_new_tokens. We only
    # ever bound by new tokens, so drop it.
    model.generation_config.max_length = None
    print("\ntype a message, or /quit to exit\n")

    history = []
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    while True:
        try:
            prompt = input("\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            break
        if prompt == "/stats":
            print_stats(manager, manifest)
            continue
        if prompt == "/reset":
            history.clear()
            print("  history cleared")
            continue
        if prompt == "/raw":
            use_chat = not use_chat
            print(f"  mode -> {'chat' if use_chat else 'raw completion'}")
            continue

        if use_chat:
            history.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        started = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                streamer=streamer,
                eos_token_id=sorted(stop_ids) or None,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - started

        generated = output.shape[1] - prompt_tokens
        reply = tokenizer.decode(
            output[0][prompt_tokens:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if use_chat:
            history.append({"role": "assistant", "content": reply})

        report = manager.report()
        print(f"\n\033[2m  {generated} tok in {elapsed:.1f}s = "
              f"{generated/elapsed:.1f} tok/s | "
              f"read {report['gb_read']:.2f} GB | "
              f"peak {peak_rss_gb():.2f} GB\033[0m\n")

    print_stats(manager, manifest)
    manager.close()


if __name__ == "__main__":
    main()
