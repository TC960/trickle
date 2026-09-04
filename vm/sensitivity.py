"""Per-layer quantization sensitivity, then mixed-precision allocation.

The depth sweep ternarized the FIRST N layers, which is a crude proxy -- it
conflates "layer index" with "layer tolerance". Measured knee: 30 layers costs
+48.6% perplexity, 45 costs +1340%. That gap says some layers are far more
fragile than others, but not WHICH.

    profile   ternarize exactly one layer at a time, measure the damage, rank
    allocate  ternarize the N most tolerant layers (by that ranking) and
              evaluate the resulting mixed-precision model properly

Profiling ranks layers by FLIP RATE and KL against the bf16 teacher, not by
perplexity. Two reasons, and they point the same way:

  * Correctness -- CLAUDE.md is explicit that perplexity is inadequate for
    judging compressed models here, because it is a geometric mean in which
    losses on some tokens cancel gains on others. Ranking which layers to
    protect is exactly the kind of decision that must not rest on it.
  * Cost -- a perplexity ranking needs a full scored pass per layer. Flip rate
    and KL need one teacher pass, cached, then one student pass per layer over
    a far smaller token budget, because a rate over 32k positions already has a
    standard error near 0.3%. Measured: ~20 hours the old way, ~30 minutes this
    way, on the same hardware.

`--metric perplexity` keeps the old behaviour for comparability. Sensitivity is
measured against the SAME teacher every time and each layer is restored before
the next, so the measurements stay independent.
"""

import argparse
import json
import time

import torch
import torch.nn as nn

from airllm_ternary.qat import ternary_ste


def layer_linears(block):
    return [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]


@torch.no_grad()
def ternarize_block(block, group_size):
    """Ternarize in place; returns the originals so it can be undone exactly."""
    saved = []
    for name, mod in layer_linears(block):
        saved.append((mod, mod.weight.detach().clone()))
        mod.weight.copy_(ternary_ste(mod.weight.float(), group_size).to(mod.weight.dtype))
    return saved


@torch.no_grad()
def restore(saved):
    for mod, w in saved:
        mod.weight.copy_(w)
    saved.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["profile", "allocate"], default="profile")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--profile-chars", type=int, default=200_000,
                    help="truncated eval for the per-layer sweep")
    ap.add_argument("--metric", choices=["kl", "perplexity"], default="kl",
                    help="what to rank layers by; kl is cheaper and is the "
                         "metric this project trusts")
    ap.add_argument("--ranking", default="/ephemeral/work/out/sensitivity.json")
    ap.add_argument("--n-ternary", type=int, default=30,
                    help="allocate mode: how many tolerant layers to ternarize")
    ap.add_argument("--layers", default=None,
                    help="allocate mode: explicit comma-separated layer list, "
                         "overriding the ranking. This exists to make the "
                         "CONTROL runnable: comparing sensitivity-ranked "
                         "selection against the old first-N depth sweep is "
                         "invalid, because that sweep used distill_seq with "
                         "200 steps of block reconstruction while this path is "
                         "pure round-to-nearest. Same code, same treatment, "
                         "only the layer choice differs.")
    ap.add_argument("--out", default="/ephemeral/work/out/mixed.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distill import find_blocks
    from perplexity import load_wikitext, perplexity

    n_gpu = torch.cuda.device_count()
    per_gpu = int(torch.cuda.get_device_properties(0).total_memory / 2**30) - 10
    mm = {i: f"{per_gpu}GiB" for i in range(n_gpu)}; mm["cpu"] = "60GiB"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, low_cpu_mem_usage=True).eval()
    blocks, _ = find_blocks(model)
    dev = next(model.parameters()).device
    print(f"  {len(blocks)} blocks", flush=True)

    if args.mode == "profile":
        ids, nb = load_wikitext(tok, limit_chars=args.profile_chars)

        if args.metric == "kl":
            from deep_eval import behaviour_delta, collect

            teacher = collect(model, ids, 2048, 1024, device=dev, verbose=False)
            n_pos = teacher["argmax"].shape[0]
            print(f"  teacher cached over {n_pos} positions\n", flush=True)

            results = []
            for i, block in enumerate(blocks):
                t0 = time.time()
                saved = ternarize_block(block, args.group_size)
                student = collect(model, ids, 2048, 1024, device=dev,
                                  verbose=False)
                restore(saved)

                flip, kl = behaviour_delta(teacher, student)
                results.append({"layer": i, "flip_rate": flip, "kl": kl,
                                "damage": kl})
                print(f"  layer {i:>2}: flips {flip*100:>6.2f}%  "
                      f"KL {kl:>9.5f}   {time.time()-t0:.0f}s", flush=True)
            baseline_desc = {"metric": "kl", "positions": int(n_pos)}
        else:
            base = perplexity(model, ids, 1024, device=dev, verbose=False,
                              n_bytes=nb)["perplexity"]
            print(f"  baseline on {ids.size(1)} tokens: {base:.4f}\n", flush=True)

            results = []
            for i, block in enumerate(blocks):
                t0 = time.time()
                saved = ternarize_block(block, args.group_size)
                pp = perplexity(model, ids, 1024, device=dev, verbose=False,
                                check_loss=False)["perplexity"]
                restore(saved)
                dmg = pp / base - 1
                results.append({"layer": i, "perplexity": pp, "damage": dmg})
                print(f"  layer {i:>2}: ppl {pp:>10.4f}  damage {dmg*100:>+9.2f}%  "
                      f"{time.time()-t0:.0f}s", flush=True)
            baseline_desc = {"metric": "perplexity", "baseline": base}

        results.sort(key=lambda r: r["damage"])
        payload = {"model": args.model, "profile_tokens": int(ids.size(1)),
                   "group_size": args.group_size, **baseline_desc,
                   "layers": results}
        with open(args.ranking, "w") as h:
            json.dump(payload, h, indent=1)

        unit = "KL" if args.metric == "kl" else "ppl damage"
        print(f"\n  most tolerant by {unit} (ternarize these first):")
        for r in results[:10]:
            print(f"    layer {r['layer']:>2}  {r['damage']:.5f}")
        print("  most sensitive (protect these):")
        for r in results[-10:]:
            print(f"    layer {r['layer']:>2}  {r['damage']:.5f}")
        print(f"\n  wrote {args.ranking}", flush=True)

    else:
        if args.layers:
            chosen = [int(x) for x in args.layers.split(",") if x.strip() != ""]
            selection = "explicit"
        else:
            rank = json.load(open(args.ranking))
            chosen = [r["layer"] for r in rank["layers"][:args.n_ternary]]
            selection = "sensitivity-ranked"
        print(f"  ternarizing {len(chosen)} layers ({selection}): "
              f"{sorted(chosen)}", flush=True)

        held = []
        for i in chosen:
            held.extend(ternarize_block(blocks[i], args.group_size))

        ids, nb = load_wikitext(tok)
        met = perplexity(model, ids, 2048, device=dev, n_bytes=nb)
        rec = {"tag": f"{args.model.split('/')[-1]}-mixed-{selection}-n{len(chosen)}",
               "model": args.model, "quant": "mixed-ternary",
               "arm": "treatment" if selection == "sensitivity-ranked" else "control",
               "pair_id": f"mixed-n{len(chosen)}",
               "n_ternary": len(chosen), "layers_ternary": sorted(chosen),
               "selection": selection, "reconstruction_steps": 0,
               "group_size": args.group_size, **met}
        print(f"\n  MIXED (t={args.n_ternary}) PERPLEXITY {met['perplexity']:.4f}  "
              f"BPB {met.get('bits_per_byte')}\n", flush=True)
        with open(args.out, "a") as h:
            h.write(json.dumps(rec) + "\n")
        with open("/ephemeral/work/out/ppl.jsonl", "a") as h:
            h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
