"""Prove that lm_head can be streamed in vocab chunks, exactly.

Claim under test: logits over a 262k vocab can be produced by loading the output
projection in row-chunks and accumulating, so peak memory is one chunk rather
than the whole 2.6 GB table -- with bit-identical results.

If true, nothing in the model is unstreamable, and the real currency is
bytes-read-per-token rather than resident bytes.
"""

import argparse
import time

import torch


def chunked_logits(hidden, weight, chunk_rows, out=None):
    """logits = hidden @ weight.T, computed chunk-by-chunk over vocab rows."""
    vocab = weight.shape[0]
    if out is None:
        out = torch.empty(*hidden.shape[:-1], vocab, dtype=torch.float32,
                          device=hidden.device)
    for start in range(0, vocab, chunk_rows):
        end = min(start + chunk_rows, vocab)
        # In a real streaming engine this slice is a disk read, not a view.
        out[..., start:end] = torch.nn.functional.linear(
            hidden, weight[start:end]).float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--hidden", type=int, default=5376)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32768)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    weight = torch.randn(args.vocab, args.hidden, dtype=torch.bfloat16, device=dev) * 0.02
    hidden = torch.randn(1, args.seq, args.hidden, dtype=torch.bfloat16, device=dev)

    full = torch.nn.functional.linear(hidden, weight).float()
    chunked = chunked_logits(hidden, weight, args.chunk)

    identical = torch.equal(full, chunked)
    max_diff = (full - chunked).abs().max().item()

    table_mb = args.vocab * args.hidden * 2 / 1e6
    chunk_mb = args.chunk * args.hidden * 2 / 1e6
    logit_mb = args.vocab * 4 / 1e6

    # Timing: how much does chunking cost in compute?
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(5):
        torch.nn.functional.linear(hidden, weight)
    torch.cuda.synchronize() if dev == "cuda" else None
    t_full = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        chunked_logits(hidden, weight, args.chunk)
    torch.cuda.synchronize() if dev == "cuda" else None
    t_chunk = (time.perf_counter() - t0) / 5

    print(f"""
  vocab {args.vocab} x hidden {args.hidden}

  BIT-IDENTICAL:        {identical}   (max|diff| = {max_diff:g})
  full table resident:  {table_mb:>8.1f} MB
  one chunk resident:   {chunk_mb:>8.1f} MB   ({args.chunk} rows)
  logit accumulator:    {logit_mb:>8.1f} MB
  peak while chunked:   {chunk_mb + logit_mb:>8.1f} MB  <- vs {table_mb:.1f} MB
  reduction:            {table_mb / (chunk_mb + logit_mb):>8.1f}x resident

  compute full:         {t_full*1000:>8.2f} ms
  compute chunked:      {t_chunk*1000:>8.2f} ms   ({t_chunk/t_full:.2f}x)
""")


if __name__ == "__main__":
    main()
