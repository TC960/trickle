"""MLP channel pruning -- the 67.5% of Gemma 4 we never touched.

Everything so far has been a PRECISION axis (fewer bits per weight). This is a
STRUCTURAL axis (fewer weights), and the two compose sub-additively: a balanced
split across axes measured +20% over quantization-only at equal compression.

Budget for reference: MLP 20.81B (67.5%) | attention 8.59B (27.9%) |
embeddings 1.41B (4.6%). Pruning 40% of MLP intermediate channels removes ~27%
of the entire model -- comparable to deleting 15 whole layers.

Gemma uses GeGLU, so gate_proj and up_proj must be pruned on the SAME output
rows and down_proj on the matching input columns, or the gating breaks. Channel
importance is measured from real activations rather than weight norms, because
what matters is how much a channel actually contributes to the output.
"""

import argparse
import json

import torch
import torch.nn as nn


@torch.no_grad()
def channel_importance(mlp, caps, block, device):
    """Mean |activation| of each intermediate channel, from real forward passes.

    Hooking down_proj's INPUT gives exactly the post-gating signal that flows
    onward -- the quantity whose removal we are pricing.
    """
    scores = None
    n = 0

    def hook(_m, inp, _o):
        nonlocal scores, n
        a = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1])
        s = a.sum(0)
        scores = s if scores is None else scores + s
        n += a.shape[0]

    h = mlp.down_proj.register_forward_hook(hook)
    try:
        for args, kwargs in caps:
            block(*args, **kwargs)
    finally:
        h.remove()
    return scores / max(n, 1)


@torch.no_grad()
def prune_to(mlp, positions):
    """Slice gate/up rows and down columns to `positions` (indices into the
    CURRENT, possibly already-pruned tensors).

    Deliberately destructive: no originals are cloned. Cloning all 60 blocks'
    MLP weights to allow restore costs a second copy of 20.81B params (~41 GB
    in bf16), which is what OOM'd the first attempt. Because the kept-channel
    sets are nested across increasing prune ratios (they all come from one
    importance order), we can prune progressively instead and never need to
    restore -- memory only ever goes down.
    """
    # GeGLU: gate and up MUST keep identical rows or the elementwise product
    # pairs the wrong channels together.
    mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[positions].contiguous()
    mlp.up_proj.weight.data = mlp.up_proj.weight.data[positions].contiguous()
    mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, positions].contiguous()
    mlp.gate_proj.out_features = len(positions)
    mlp.up_proj.out_features = len(positions)
    mlp.down_proj.in_features = len(positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--ratios", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", default="/ephemeral/work/out/mlp_prune.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deep_eval import behaviour_delta, collect
    from distill import find_blocks, _move, _SKIP_KWARGS
    from perplexity import load_wikitext, perplexity

    tok = AutoTokenizer.from_pretrained(args.model)
    n_gpu = torch.cuda.device_count()
    # A hardcoded 45GiB was sized for 2x80GB. On a single card it forces ~17 GB
    # of Gemma 4 onto DISK, which is both slow and a silent correctness risk.
    # Leave ~10GiB per card for activations and spill to host RAM, never disk.
    per_gpu = int(torch.cuda.get_device_properties(0).total_memory / 2**30) - 10
    mm = {i: f"{per_gpu}GiB" for i in range(n_gpu)}; mm["cpu"] = "60GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, low_cpu_mem_usage=True).eval()
    blocks, _ = find_blocks(model)
    dev0 = next(model.parameters()).device

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    L = args.seq_len
    calib = [ids[i*L:(i+1)*L].unsqueeze(0).to(dev0) for i in range(args.n_calib)]

    caps = [[] for _ in blocks]
    handles = []
    def mk(i):
        def hook(_m, a, kw):
            caps[i].append((a, {k: v for k, v in kw.items() if k not in _SKIP_KWARGS}))
        return hook
    for i, b in enumerate(blocks):
        handles.append(b.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.no_grad():
        for b in calib:
            model(b, use_cache=False)
    for h in handles:
        h.remove()
    print(f"  captured inputs for {len(blocks)} blocks", flush=True)

    inter = blocks[0].mlp.gate_proj.out_features
    print(f"  intermediate size {inter}", flush=True)

    print("  scoring channels...", flush=True)
    keep_order = []
    for i, b in enumerate(blocks):
        imp = channel_importance(b.mlp, caps[i], b, dev0)
        # Rank on CPU: 60 x 21504 int64 is trivial there, and the captured
        # activations below are what we actually need the GPU memory for.
        keep_order.append(torch.argsort(imp, descending=True).cpu())
        caps[i] = None          # ~10.5 GB of captured hidden states, freed as we go
    del caps
    torch.cuda.empty_cache()
    print("  scored all blocks", flush=True)

    eval_ids, nb = load_wikitext(tok)

    # Behavioural reference FIRST. Pruning is judged on how often it changes
    # the model's token choice, not on a geometric mean in which losses on some
    # tokens cancel gains on others. Perplexity is still recorded, last, for
    # comparability with published pruning work.
    behav_ids, _ = load_wikitext(tok, limit_chars=400_000)
    teacher = collect(model, behav_ids, 2048, 1024, device=dev0, verbose=False)
    print(f"  teacher cached over {teacher['argmax'].shape[0]} positions",
          flush=True)

    base = perplexity(model, eval_ids, 2048, device=dev0, verbose=False,
                      n_bytes=nb)
    print(f"  baseline ppl {base['perplexity']:.4f}\n", flush=True)

    # Ascending order matters: kept-channel sets are nested, so each ratio can
    # be reached by pruning further from the previous one.
    ratios = sorted(float(r) for r in args.ratios.split(","))
    # Original channel indices still present in each block, ascending.
    current = [torch.arange(inter) for _ in blocks]

    for ratio in ratios:
        keep_n = int(inter * (1 - ratio))
        for i, b in enumerate(blocks):
            target = torch.sort(keep_order[i][:keep_n]).values
            # Where those original indices sit inside the current tensors.
            pos = torch.searchsorted(current[i], target)
            prune_to(b.mlp, pos.to(dev0))
            current[i] = target
        torch.cuda.empty_cache()

        student = collect(model, behav_ids, 2048, 1024, device=dev0,
                          verbose=False)
        flip, kl = behaviour_delta(teacher, student)
        met = perplexity(model, eval_ids, 2048, device=dev0, verbose=False,
                         n_bytes=nb)

        # MLP is 67.5% of params; pruning `ratio` of it removes 0.675*ratio.
        rec = {"tag": f"gemma-mlpprune-{int(ratio*100)}",
               "model": args.model, "ratio": ratio,
               "channels": f"{inter}->{keep_n}",
               "model_params_removed_pct": round(67.5 * ratio, 1),
               "flip_rate": flip, "kl_mean": kl,
               "perplexity": met["perplexity"],
               "bits_per_byte": met.get("bits_per_byte"),
               "delta_pct": round((met["perplexity"]/base["perplexity"]-1)*100, 2)}
        print(f"  prune {ratio*100:>4.0f}% of MLP ({inter}->{keep_n}) = "
              f"{rec['model_params_removed_pct']}% of model:  "
              f"flips {flip*100:>6.2f}%  KL {kl:>8.5f}  "
              f"ppl {met['perplexity']:.4f} ({rec['delta_pct']:+.2f}%)", flush=True)
        with open(args.out, "a") as h:
            h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
