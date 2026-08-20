"""Block-wise quantization-aware reconstruction.

Full end-to-end QAT on a 31B model needs ~375 GB of optimizer state. This does
the tractable thing instead: optimize one decoder block at a time so that its
*output* matches the unquantized teacher block's output on calibration data,
with gradients flowing through a straight-through estimator into latent weights.

    for each block i:
        target = teacher_block_i(captured_input_i)      # unquantized reference
        train student_i (ternary via STE) to match target
        export packed codes, free the block

Peak memory is one block's worth of weights, gradients and optimizer state --
the same bounded-footprint invariant the inference engine enforces.

Two properties make this work:

  teacher forcing   every block's input is captured from the *clean* bf16 model,
                    so quantization error cannot compound across depth and each
                    block is an independent problem
  local objective   we optimize output fidelity, which is what actually matters,
                    rather than weight-space similarity, which does not
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat import QATLinear, swap_to_qat, ternary_stats


# Stateful/cache kwargs that must not be captured or replayed.
_CACHE_KWARGS = frozenset(
    {"past_key_value", "past_key_values", "use_cache", "cache_position", "layer_idx"}
)


def _to_device(obj, device):
    """Recursively move tensors inside args/kwargs structures."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [_to_device(item, device) for item in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    return obj


class BlockIOCapture:
    """Records the exact inputs each decoder block receives from the clean model.

    Capturing kwargs rather than reconstructing them matters: Gemma 4 hands
    different rotary embeddings and mask types to sliding vs full-attention
    layers, and reproducing that by hand is how subtle bugs get in.
    """

    def __init__(self, blocks, store_device="cpu"):
        self.blocks = blocks
        self.store_device = store_device
        self.captured = [[] for _ in blocks]
        self._handles = []

    def __enter__(self):
        for index, block in enumerate(self.blocks):
            def hook(_module, args, kwargs, _index=index):
                # Drop cache-related kwargs. A Cache object is stateful and gets
                # filled during capture, so replaying it later would present a
                # key length of cached+new against a mask sized for new only.
                clean = {
                    key: value for key, value in kwargs.items()
                    if key not in _CACHE_KWARGS
                }
                self.captured[_index].append(
                    (_to_device(args, self.store_device),
                     _to_device(clean, self.store_device))
                )

            self._handles.append(
                block.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


@torch.no_grad()
def capture_calibration(model, blocks, batches, store_device="cpu"):
    """Run the clean model over calibration batches, recording per-block inputs."""
    with BlockIOCapture(blocks, store_device) as capture:
        for batch in batches:
            # use_cache=False keeps the run stateless, so each captured input is
            # a pure function of the calibration batch and can be replayed.
            model(**batch, use_cache=False)
    return capture.captured


def _block_output(block, args, kwargs):
    """Call a decoder block and return just its hidden-state output."""
    output = block(*args, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def reconstruct_block(
    teacher_block: nn.Module,
    captured,
    *,
    group_size: int = 128,
    steps: int = 200,
    lr: float = 1e-4,
    device: str = "mps",
    log_every: int = 50,
    verbose: bool = True,
):
    """Train a ternary copy of one block to match its teacher's outputs.

    Returns (student_block, {module_path: QATLinear}, metrics).
    """
    teacher_block = teacher_block.to(device).eval()
    for param in teacher_block.parameters():
        param.requires_grad_(False)

    student_block = copy.deepcopy(teacher_block)
    qat_modules = swap_to_qat(student_block, group_size)
    student_block = student_block.to(device).train()

    optimizer = torch.optim.AdamW(
        [m.latent_weight for m in qat_modules.values()], lr=lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Precompute teacher targets once; they never change during this block's
    # training and recomputing them every step would double the cost.
    targets = []
    with torch.no_grad():
        for args, kwargs in captured:
            args_d, kwargs_d = _to_device(args, device), _to_device(kwargs, device)
            targets.append(_block_output(teacher_block, args_d, kwargs_d).detach())

    baseline = None
    history = []
    started = time.time()

    for step in range(steps):
        batch_index = step % len(captured)
        args, kwargs = captured[batch_index]
        args_d, kwargs_d = _to_device(args, device), _to_device(kwargs, device)
        target = targets[batch_index]

        output = _block_output(student_block, args_d, kwargs_d)
        # Normalize by target energy. Raw MSE grows with depth because
        # activation magnitude does, so a fixed learning rate under-trains
        # shallow blocks and oscillates on deep ones. Dividing by the target's
        # mean square makes the gradient scale comparable across all blocks.
        target_energy = target.float().pow(2).mean().clamp_min(1e-8)
        loss = F.mse_loss(output.float(), target.float()) / target_energy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [m.latent_weight for m in qat_modules.values()], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        if baseline is None:
            baseline = loss.item()
        if verbose and (step % log_every == 0 or step == steps - 1):
            relative = loss.item() / baseline if baseline > 0 else 0.0
            print(f"      step {step:>4}  mse {loss.item():.6f}  "
                  f"({relative:.3f} of initial)")
        history.append(loss.item())

    # Final fidelity against the teacher, the number that actually matters.
    with torch.no_grad():
        cosines = []
        for (args, kwargs), target in zip(captured, targets):
            args_d, kwargs_d = _to_device(args, device), _to_device(kwargs, device)
            output = _block_output(student_block, args_d, kwargs_d)
            cosines.append(
                F.cosine_similarity(
                    output.float().flatten(), target.float().flatten(), dim=0
                ).item()
            )

    metrics = {
        "initial_mse": baseline,
        "final_mse": history[-1],
        "mse_reduction": baseline / history[-1] if history[-1] > 0 else float("inf"),
        "output_cosine": sum(cosines) / len(cosines),
        "seconds": round(time.time() - started, 1),
        "codes": ternary_stats(next(iter(qat_modules.values()))),
    }
    return student_block, qat_modules, metrics


@torch.no_grad()
def measure_naive_baseline(teacher_block, captured, group_size, device):
    """Output fidelity from plain absmean rounding, with no training.

    This is the PTQ number the reconstruction has to beat; without it there is
    no evidence the training did anything.
    """
    from .qat import ternary_ste

    student = copy.deepcopy(teacher_block).to(device).eval()
    for module in student.modules():
        if isinstance(module, nn.Linear):
            module.weight.copy_(ternary_ste(module.weight.float(), group_size))

    cosines = []
    for args, kwargs in captured:
        args_d, kwargs_d = _to_device(args, device), _to_device(kwargs, device)
        target = _block_output(teacher_block.to(device), args_d, kwargs_d)
        output = _block_output(student, args_d, kwargs_d)
        cosines.append(
            F.cosine_similarity(
                output.float().flatten(), target.float().flatten(), dim=0
            ).item()
        )
    return sum(cosines) / len(cosines)


def reconstruct_model(
    model,
    blocks,
    batches,
    *,
    group_size: int = 128,
    steps: int = 200,
    lr: float = 1e-4,
    device: str = "mps",
    compare_baseline: bool = True,
    verbose: bool = True,
):
    """Reconstruct every block in sequence. Returns per-block metrics."""
    if verbose:
        print(f"capturing calibration inputs for {len(blocks)} blocks...")
    captured = capture_calibration(model, blocks, batches)

    results = []
    for index, block in enumerate(blocks):
        if verbose:
            print(f"\n  block {index}/{len(blocks)-1}")

        naive = None
        if compare_baseline:
            naive = measure_naive_baseline(
                block, captured[index], group_size, device
            )
            if verbose:
                print(f"      PTQ baseline cosine: {naive:.4f}")

        _, qat_modules, metrics = reconstruct_block(
            block, captured[index],
            group_size=group_size, steps=steps, lr=lr,
            device=device, verbose=verbose,
        )
        metrics["block"] = index
        metrics["naive_cosine"] = naive
        if naive is not None:
            metrics["cosine_gain"] = metrics["output_cosine"] - naive
        results.append(metrics)

        if verbose:
            gain = f" (+{metrics['cosine_gain']:.4f} over PTQ)" if naive else ""
            print(f"      trained cosine: {metrics['output_cosine']:.4f}{gain}")

        # Free the block's optimizer state before moving on; this is what keeps
        # peak memory at one block rather than the whole model.
        del qat_modules
        if device == "mps":
            torch.mps.empty_cache()

    return results
