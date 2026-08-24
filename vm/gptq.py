"""GPTQ-style error-compensated quantization.

Round-to-nearest treats every weight independently, so each rounding error is
simply absorbed as loss. GPTQ instead quantizes column by column and pushes the
error from column j into the columns not yet quantized, weighted by the inverse
Hessian H^-1 = (2 X X^T + lambda I)^-1 built from real layer inputs. Later
columns then compensate for earlier mistakes.

The gap this closes is not small. On LLaMA-30B at 2-bit, plain rounding gives
perplexity 13.01 against fp16 4.10; error compensation is what makes low-bit
usable at all.

Used here as INITIALIZATION -- GPTQ first, then the existing STE training
refines. That is the same order GSQ uses, and it starts the optimizer from a far
better point than round-to-nearest does.
"""

import math

import torch
import torch.nn as nn


class HessianAccumulator:
    """Accumulates H = 2 X X^T over calibration batches for one linear layer."""

    def __init__(self, in_features, device):
        self.H = torch.zeros((in_features, in_features), dtype=torch.float32,
                             device=device)
        self.n = 0

    def add(self, x):
        # x: [..., in_features] -> flatten every token into a row
        x = x.reshape(-1, x.shape[-1]).to(torch.float32)
        b = x.shape[0]
        # Running mean so batches of different sizes weight correctly.
        self.H *= self.n / (self.n + b)
        self.n += b
        self.H += (2.0 / self.n) * (x.t() @ x)


@torch.no_grad()
def gptq_quantize(weight, H, bits=3, group_size=128, percdamp=0.01,
                  actorder=True):
    """Quantize [out, in] with Hessian-weighted error feedback.

    actorder: process columns in descending diagonal-Hessian order, so the
    most-influential inputs are quantized while the most compensation budget
    remains. This is worth a lot at low bit-widths.
    """
    W = weight.detach().clone().float()
    out_f, in_f = W.shape
    dev = W.device

    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    # Dampen the diagonal so the Cholesky is well conditioned.
    damp = percdamp * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp

    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    Q = torch.zeros_like(W)
    qmax = 2 ** bits - 1

    for start in range(0, in_f, group_size):
        end = min(start + group_size, in_f)
        W1 = W[:, start:end].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[start:end, start:end]

        # One scale/zero per group, fixed from the group's pre-quant range.
        lo = W1.min(dim=1, keepdim=True).values
        hi = W1.max(dim=1, keepdim=True).values
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        zero = torch.round(-lo / scale)

        for j in range(end - start):
            w = W1[:, j]
            d = Hinv1[j, j]
            q = torch.clamp(torch.round(w.unsqueeze(1) / scale) + zero, 0, qmax)
            dq = ((q - zero) * scale).squeeze(1)
            Q1[:, j] = dq
            # Error from this column, spread over the remaining ones.
            err = (w - dq) / d
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            E1[:, j] = err

        Q[:, start:end] = Q1
        # Propagate this group's accumulated error into all later columns.
        if end < in_f:
            W[:, end:] -= E1 @ Hinv[start:end, end:]

    if actorder:
        Q = Q[:, invperm]
    return Q.to(weight.dtype)


def collect_hessians(block, inputs_iter, device):
    """Hook every Linear in a block and accumulate its input Hessian."""
    accs, handles = {}, []

    def mk(name, mod):
        def hook(_m, inp, _out):
            x = inp[0]
            if name not in accs:
                accs[name] = HessianAccumulator(x.shape[-1], device)
            accs[name].add(x.detach())
        return hook

    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(mk(name, mod)))
    try:
        for args, kwargs in inputs_iter:
            block(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()
    return accs


@torch.no_grad()
def gptq_init_block(block, accs, bits, group_size, verbose=False):
    """Replace each Linear's weight with its GPTQ-quantized version, in place."""
    done = 0
    for name, mod in block.named_modules():
        if not isinstance(mod, nn.Linear) or name not in accs:
            continue
        H = accs[name].H
        q = gptq_quantize(mod.weight, H, bits=bits, group_size=group_size)
        if verbose:
            rel = ((mod.weight.float() - q.float()).norm()
                   / mod.weight.float().norm()).item()
            print(f"      gptq {name:<22} rel_err {rel:.4f}", flush=True)
        mod.weight.copy_(q)
        done += 1
    return done
