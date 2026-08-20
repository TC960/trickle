"""Locate where the streamed model diverges from the reference.

Compares, in order: tied-weight identity, one linear layer's output, per-layer
hidden states, and final logits. The first thing that differs is the bug.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from airllm_ternary.model import load_streaming_model


def report(label, a, b):
    a32, b32 = a.detach().float(), b.detach().float()
    if a32.shape != b32.shape:
        print(f"  {label:<34} SHAPE MISMATCH {tuple(a32.shape)} vs {tuple(b32.shape)}")
        return False
    exact = torch.equal(a32, b32)
    max_abs = (a32 - b32).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        a32.flatten(), b32.flatten(), dim=0
    ).item()
    flag = "OK  " if exact else ("close" if max_abs < 1e-2 else "DIFF")
    print(f"  {label:<34} {flag}  max|d|={max_abs:.6g}  cos={cos:.6f}")
    return exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--shards", required=True)
    parser.add_argument("--device", default="cpu", help="cpu for determinism")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer("The capital of France is", return_tensors="pt").to(args.device)

    print("loading reference...")
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(args.device).eval()

    print("loading streamed...")
    ours, manager = load_streaming_model(
        args.model, args.shards,
        budget_gb=8.0, device=args.device, prefetch=False,
    )

    # --- 1. structural checks -------------------------------------------------
    print("\n=== structure ===")
    ref_emb = ref.get_input_embeddings().weight
    our_emb = ours.get_input_embeddings().weight
    print(f"  ref  embed device={ref_emb.device} dtype={ref_emb.dtype}")
    print(f"  ours embed device={our_emb.device} dtype={our_emb.dtype}")

    ref_head = ref.get_output_embeddings()
    our_head = ours.get_output_embeddings()
    for label, head, emb in (("ref", ref_head, ref_emb), ("ours", our_head, our_emb)):
        if head is None:
            print(f"  {label} lm_head: None")
            continue
        weight = head.weight
        print(f"  {label} lm_head: device={weight.device} "
              f"is_meta={weight.is_meta} tied_to_embed={weight is emb}")

    report("embed_tokens", ref_emb, our_emb)

    # Run one forward so the residency manager has loaded every layer; module
    # internals are None until their shard is fetched.
    with torch.inference_mode():
        ours(**ids, use_cache=False)

    # --- 2. one linear in isolation ------------------------------------------
    print("\n=== single linear (layer 0 q_proj) ===")
    ref_q = ref.model.layers[0].self_attn.q_proj
    our_q = ours.model.layers[0].self_attn.q_proj
    print(f"  ref type={type(ref_q).__name__} ours type={type(our_q).__name__}")
    print(f"  ref weight_scale={ref_q.weight_scale.float().item():.8g}")
    print(f"  ours scale={our_q.scale.float().item():.8g}")
    report("q_proj codes", ref_q.weight, our_q.dequantized())

    torch.manual_seed(0)
    x = torch.randn(1, 4, ref_q.in_features, dtype=torch.bfloat16, device=args.device)
    with torch.inference_mode():
        report("q_proj output", ref_q(x), our_q(x))

    # --- 3. norms ------------------------------------------------------------
    print("\n=== layer 0 norms ===")
    for name in ("input_layernorm", "post_attention_layernorm"):
        report(name, getattr(ref.model.layers[0], name).weight,
               getattr(ours.model.layers[0], name).weight)
    report("attn_sub_norm",
           ref.model.layers[0].self_attn.attn_sub_norm.weight,
           ours.model.layers[0].self_attn.attn_sub_norm.weight)
    report("mlp ffn_sub_norm",
           ref.model.layers[0].mlp.ffn_sub_norm.weight,
           ours.model.layers[0].mlp.ffn_sub_norm.weight)
    report("model.norm", ref.model.norm.weight, ours.model.norm.weight)

    # --- 4. per-layer hidden states ------------------------------------------
    print("\n=== hidden states per layer ===")
    with torch.inference_mode():
        ref_out = ref(**ids, output_hidden_states=True, use_cache=False)
        our_out = ours(**ids, output_hidden_states=True, use_cache=False)

    first_bad = None
    for index, (a, b) in enumerate(zip(ref_out.hidden_states, our_out.hidden_states)):
        exact = report(f"hidden[{index}]", a, b)
        if not exact and first_bad is None:
            first_bad = index

    print("\n=== logits ===")
    report("logits", ref_out.logits, our_out.logits)
    print(f"  ref  argmax: {ref_out.logits[0, -1].argmax().item()} "
          f"({tokenizer.decode([ref_out.logits[0, -1].argmax().item()])!r})")
    print(f"  ours argmax: {our_out.logits[0, -1].argmax().item()} "
          f"({tokenizer.decode([our_out.logits[0, -1].argmax().item()])!r})")

    if first_bad is not None:
        print(f"\n>>> first divergent hidden state: index {first_bad}")
        print(">>> hidden_states[0] is the embedding output; "
              "index N is the output of decoder layer N-1")

    manager.close()


if __name__ == "__main__":
    main()
