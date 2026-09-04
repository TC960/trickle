"""End-to-end ternary distillation: freeze the codes, train the scales on logit KL.

WHY THIS EXISTS
Per-block reconstruction failed, and not because the metric was bad -- because
the OBJECTIVE was. Minimizing each block's own output error is a proxy for what
matters (the final logits), and any per-block proxy is blind to error compounding
by construction. Measured: ~16% relative error per block, cosine 0.986, and
end-to-end perplexity 222,296 against a 5.19 baseline.

WHAT THIS DOES DIFFERENTLY
Trains against the full-precision teacher's LOGITS. Drift cannot hide from a loss
that sees the model's actual output.

WHY IT FITS IN MEMORY
Full QAT on 31B needs ~375 GB of optimizer state. Here the ternary codes are
FROZEN int8 buffers and only the per-group scales are trainable:
    29.4B weights / group 128 = ~230M trainable params  (LoRA scale)
Codes at int8 are 29.4 GB -- smaller than the bf16 model -- and split across two
GPUs. Optimizer state is then ~4 GB rather than ~375 GB.

    weight = codes.to(dtype) * scale        # scale is the only Parameter
"""

import argparse
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLearned(nn.Module):
    """Ternary weights with a learnable SCALE and a learnable THRESHOLD.

    Why the threshold matters, and why scale-only training plateaus:

    A scale rescales all three levels {-1, 0, +1} together. It cannot change
    WHICH weights land on zero. For a 3-level quantizer, that assignment is
    arguably the more important parameter -- it sets the sparsity pattern.
    PV-Tuning Table 1 measures the consequence on 2-bit scalar quantization:
    training continuous parameters alone gets perplexity from 3290 to 16.77 and
    stops, while methods that also move the discrete assignment reach 8.5.

    So `mu` is learnable here: the zero band is |w|/scale < mu. Training uses a
    soft ternarization whose sharpness anneals upward, with a straight-through
    estimator so the forward pass stays exactly ternary:

        soft = sigmoid(s*(x - mu)) - sigmoid(-s*(x + mu))     differentiable
        hard = sign(x) * 1[|x| > mu]                          exactly ternary
        q    = soft + (hard - soft).detach()                  fwd hard, bwd soft

    Both `scale` and `mu` receive gradient; the code assignment itself moves as
    `mu` moves, which is the degree of freedom scale-only training lacks.
    """

    def __init__(self, linear: nn.Linear, group_size: int = 128,
                 init_mu: float = 0.5):
        super().__init__()
        w = linear.weight.data
        self.out_features, self.in_features = w.shape
        self.group_size = group_size
        dev, dtype = w.device, w.dtype

        pad = (-self.in_features) % group_size
        wp = F.pad(w.float(), (0, pad)) if pad else w.float()
        self.padded_in = wp.shape[1]
        groups = wp.reshape(-1, group_size)

        scale = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        # Normalized weights; the frozen part is the SIGN, not the assignment.
        self.register_buffer("wn", (groups / scale).to(torch.float16)
                             .reshape(self.out_features, self.padded_in),
                             persistent=False)
        self.scale = nn.Parameter(scale.reshape(self.out_features, -1)
                                  .to(torch.float32).to(dev))
        # One threshold per group, same granularity as the scale.
        self.mu = nn.Parameter(torch.full_like(self.scale, float(init_mu)))
        self.bias = linear.bias
        self.compute_dtype = dtype
        self.sharpness = 8.0  # annealed upward during training

    def _quant(self):
        x = self.wn.to(torch.float32).reshape(-1, self.group_size)
        mu = self.mu.reshape(-1, 1).abs()
        s = self.sharpness
        soft = torch.sigmoid(s * (x - mu)) - torch.sigmoid(-s * (x + mu))
        hard = torch.sign(x) * (x.abs() > mu).to(x.dtype)
        q = soft + (hard - soft).detach()
        w = (q * self.scale.reshape(-1, 1)).reshape(
            self.out_features, self.padded_in)
        return w[:, :self.in_features]

    def forward(self, x):
        w = self._quant()
        return F.linear(x, w.to(x.dtype), self.bias)

    @torch.no_grad()
    def sparsity(self):
        x = self.wn.to(torch.float32).reshape(-1, self.group_size)
        return (x.abs() <= self.mu.reshape(-1, 1).abs()).float().mean().item()


def convert(model, blocks, group_size, skip_first, skip_last, verbose=True):
    """Replace Linears in the eligible blocks with TernaryScaled."""
    n = len(blocks)
    swapped, params = 0, []
    for i, block in enumerate(blocks):
        if i < skip_first or i >= n - skip_last:
            continue
        for path, mod in list(block.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            new = TernaryLearned(mod, group_size)
            parent, _, attr = path.rpartition(".")
            tgt = block
            if parent:
                for part in parent.split("."):
                    tgt = getattr(tgt, part)
            setattr(tgt, attr, new)
            params.extend([new.scale, new.mu])
            swapped += 1
            # Release the bf16 original immediately; otherwise it coexists with
            # the new int8 codes across the whole conversion and doubles peak.
            mod.weight = None
            del mod
            gc.collect(); torch.cuda.empty_cache()
        gc.collect(); torch.cuda.empty_cache()
    if verbose:
        total = sum(p.numel() for p in params)
        print(f"  converted {swapped} linears in blocks "
              f"[{skip_first}, {n-skip_last}) -> {total/1e6:.1f}M trainable scales",
              flush=True)
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--skip-last", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distill import find_blocks

    print(f"=== end-to-end scale distillation: {args.model} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    n_gpu = torch.cuda.device_count()
    mm = {i: "42GiB" for i in range(n_gpu)}; mm["cpu"] = "0GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, low_cpu_mem_usage=True).eval()
    blocks, path = find_blocks(model)
    print(f"  {len(blocks)} blocks at {path}", flush=True)

    # Teacher logits, captured BEFORE conversion, on the same batches.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    L, n_seq = args.seq_len, args.steps
    stride = max(1, (ids.numel() - L) // max(n_seq, 1))
    batches = [ids[k * stride: k * stride + L].unsqueeze(0)
               for k in range(n_seq) if ids[k * stride: k * stride + L].numel() == L]
    print(f"  {len(batches)} training sequences x {L} tokens", flush=True)

    dev0 = next(model.parameters()).device
    TOPK = 128
    print(f"  capturing teacher top-{TOPK} logprobs...", flush=True)
    teacher = []
    with torch.no_grad():
        for i, b in enumerate(batches):
            lp = F.log_softmax(
                model(b.to(dev0), use_cache=False).logits[0].float(), dim=-1)
            v, ix = lp.topk(TOPK, dim=-1)
            teacher.append((v.to(torch.float16).cpu(), ix.to(torch.int32).cpu()))
            del lp
            if i % 50 == 0:
                torch.cuda.empty_cache()
                print(f"    {i}/{len(batches)}", flush=True)
    mb = sum(v.numel() * 2 + ix.numel() * 4 for v, ix in teacher) / 1e6
    print(f"  captured {len(teacher)} x top-{TOPK} ({mb:.0f} MB on CPU)", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    scales = convert(model, blocks, args.group_size, args.skip_first, args.skip_last)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in scales:
        p.requires_grad_(True)

    opt = torch.optim.AdamW(scales, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("  training scales on logit KL...", flush=True)
    t0, first = time.time(), None
    mods = [m for m in model.modules() if isinstance(m, TernaryLearned)]
    for step in range(args.steps):
        # Soft early (gradient flows through the assignment), hard late (the
        # forward pass is always exactly ternary via the STE either way).
        frac = step / max(args.steps - 1, 1)
        for m in mods:
            m.sharpness = 8.0 + frac * 40.0
        b = batches[step % len(batches)].to(dev0)
        tgt = teacher[step % len(teacher)]

        t_lp, t_ix = tgt
        t_lp = t_lp.to(dev0).float()
        t_ix = t_ix.to(dev0).long()

        logits = model(b, use_cache=False).logits[0].float()
        s_lp = F.log_softmax(logits, dim=-1)
        # KL(teacher || student) restricted to the teacher's top-K support --
        # that is where essentially all the mass is, and it avoids materializing
        # a full-vocab distribution for both models at once.
        s_at = s_lp.gather(-1, t_ix)
        p = t_lp.exp()
        loss = (p * (t_lp - s_at)).sum(-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scales, 1.0)
        opt.step(); sched.step()
        if first is None:
            first = loss.item()
        if step % 20 == 0 or step == args.steps - 1:
            sp = mods[0].sparsity() if mods else 0.0
            print(f"    step {step:>4}  KL {loss.item():.6f} zeros={sp:.3f} "
                  f"({loss.item()/first:.3f} of initial)  "
                  f"{(time.time()-t0)/max(step,1):.1f}s/step", flush=True)
        del logits, loss

    print("\n  evaluating end to end...", flush=True)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    from perplexity import load_wikitext, perplexity
    eids, nb = load_wikitext(tok)
    met = perplexity(model, eids, 2048, device=dev0, n_bytes=nb)

    tag = args.tag or f"{args.model.split('/')[-1]}-ternary-e2e-scales"
    rec = {"tag": tag, "model": args.model, "quant": "ternary-e2e-scales",
           "group_size": args.group_size, "steps": args.steps, "lr": args.lr,
           "skip_first": args.skip_first, "skip_last": args.skip_last,
           "trainable_params": sum(p.numel() for p in scales),
           "learnable_threshold": True, **met}
    print(f"\n  {tag} PERPLEXITY {met['perplexity']:.4f}  BPB {met.get('bits_per_byte')}\n",
          flush=True)
    with open("/ephemeral/work/out/ppl.jsonl", "a") as h:
        h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
