"""Validate GPTQ against round-to-nearest using REAL activations.

Synthetic activations cannot settle this: GPTQ's advantage comes from the
correlation and outlier structure of genuine transformer activations, which
random tensors do not have. Verified so far -- with H=I the implementation
reproduces round-to-nearest exactly, so the algorithm is right; what was missing
was a well-conditioned Hessian from real data.

Calibration and evaluation use DISJOINT text, so any gain is generalization and
not Hessian overfitting.
"""

import argparse
import json

import torch
import torch.nn as nn

from gptq import HessianAccumulator, gptq_quantize
from airllm_ternary.qat import uniform_ste


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--group", default="128,64")
    ap.add_argument("--calib-tokens", type=int, default=65536)
    ap.add_argument("--out", default="/ephemeral/work/out/gptq_real.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from distill import find_blocks, _move, _SKIP_KWARGS

    tok = AutoTokenizer.from_pretrained(args.model)
    n_gpu = torch.cuda.device_count()
    mm = {i: "45GiB" for i in range(n_gpu)}; mm["cpu"] = "0GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, low_cpu_mem_usage=True).eval()
    blocks, _ = find_blocks(model)
    block = blocks[args.layer]
    dev = next(block.parameters()).device

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    L = 2048
    n_seq = max(2, args.calib_tokens // L)
    calib = [ids[i*L:(i+1)*L].unsqueeze(0) for i in range(n_seq)]
    # Disjoint eval text, drawn from far away in the corpus.
    off = n_seq * L + 10_000
    evals = [ids[off + i*L: off + (i+1)*L].unsqueeze(0) for i in range(8)]
    print(f"  calib {len(calib)}x{L} = {len(calib)*L} tokens; eval {len(evals)}x{L}",
          flush=True)

    # Capture this block's inputs (hidden states + kwargs) from a real forward.
    caps = []
    h = block.register_forward_pre_hook(
        lambda _m, a, kw: caps.append((_move(a, dev),
            {k: _move(v, dev) for k, v in kw.items() if k not in _SKIP_KWARGS})),
        with_kwargs=True)
    with torch.no_grad():
        for b in calib + evals:
            model(b.to(dev), use_cache=False)
    h.remove()
    calib_caps, eval_caps = caps[:len(calib)], caps[len(calib):]
    print(f"  captured {len(caps)} block inputs", flush=True)

    # Real Hessians per Linear, from calibration inputs only.
    accs, handles = {}, []
    def mk(name):
        def hook(_m, inp, _out):
            x = inp[0]
            if name not in accs:
                accs[name] = HessianAccumulator(x.shape[-1], dev)
            accs[name].add(x.detach())
        return hook
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        for a, kw in calib_caps:
            block(*a, **kw)
    for hh in handles:
        hh.remove()
    print(f"  Hessians for {len(accs)} linears", flush=True)

    # Reference outputs of the untouched block on held-out inputs.
    with torch.no_grad():
        ref = [block(*a, **kw)[0].float() for a, kw in eval_caps]

    def block_err():
        with torch.no_grad():
            num = den = 0.0
            for (a, kw), r in zip(eval_caps, ref):
                o = block(*a, **kw)[0].float()
                num += (o - r).pow(2).sum().item(); den += r.pow(2).sum().item()
        return (num / den) ** 0.5

    originals = {n: m.weight.detach().clone()
                 for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    results = []
    for bits in [int(x) for x in args.bits.split(",")]:
        for g in [int(x) for x in args.group.split(",")]:
            row = {"bits": bits, "group": g, "layer": args.layer}
            for method in ("rtn", "gptq"):
                with torch.no_grad():
                    for n, m in block.named_modules():
                        if not isinstance(m, nn.Linear):
                            continue
                        W = originals[n]
                        q = (uniform_ste(W.float(), bits, g) if method == "rtn"
                             else gptq_quantize(W, accs[n].H, bits=bits,
                                                group_size=g, actorder=True))
                        m.weight.copy_(q.to(m.weight.dtype))
                row[method] = block_err()
            with torch.no_grad():
                for n, m in block.named_modules():
                    if isinstance(m, nn.Linear):
                        m.weight.copy_(originals[n])
            row["gain_pct"] = (1 - row["gptq"] / row["rtn"]) * 100
            results.append(row)
            print(f"  bits={bits} g={g}:  RTN {row['rtn']:.4f}   "
                  f"GPTQ {row['gptq']:.4f}   gain {row['gain_pct']:+.1f}%",
                  flush=True)

    with open(args.out, "a") as fh:
        for r in results:
            fh.write(json.dumps({**r, "model": args.model}) + "\n")


if __name__ == "__main__":
    main()
