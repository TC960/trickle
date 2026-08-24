# Evaluate published low-bit checkpoints at OUR scale with OUR harness.
#
# Why this matters more than another training run: the literature says 2-bit at
# 30B works (QTIP LLaMA-30B 4.83 vs 4.10 fp16, +17.8%) and ternary at 32B retains
# 93% (CAT-Q Qwen3-32B). Both claims are testable in an hour by downloading
# someone else's finished work instead of spending GPU-hours reproducing it.
set -uo pipefail
W=/ephemeral/work
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false
PY=$W/venv/bin/python
L=$W/logs
cd $W/code

echo "[$(date +%H:%M)] installing GuidedQuant any-precision loader"
export VIRTUAL_ENV=$W/venv PATH="$HOME/.local/bin:$PATH"
uv pip install -q ap-gemv -i https://jinukkim.me/whl/cu124 2>&1 | tail -2 \
  || pip install -q ap-gemv -i https://jinukkim.me/whl/cu124 2>&1 | tail -2 \
  || echo "  ap-gemv install failed - will try plain load"
uv pip install -q any-precision-llm 2>&1 | tail -1 || true

echo "[$(date +%H:%M)] downloading gemma-3-27b-it 2-bit GuidedQuant"
$PY -c "
from huggingface_hub import snapshot_download
p=snapshot_download('jusjinuk/gemma-3-27b-it-2bit-GuidedQuant-LNQ')
print('  ->', p)
" 2>&1 | tail -2

echo "[$(date +%H:%M)] attempting load + perplexity"
$PY - <<'PY' 2>&1 | tail -30
import torch, json, traceback
from transformers import AutoTokenizer
M = "jusjinuk/gemma-3-27b-it-2bit-GuidedQuant-LNQ"
tok = AutoTokenizer.from_pretrained(M)
model = None
for how in ("any_precision", "transformers"):
    try:
        if how == "any_precision":
            from any_precision import AnyPrecisionForCausalLM
            model = AnyPrecisionForCausalLM.from_quantized(M, precisions=[2],
                                                           trust_remote_code=True)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                M, dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, low_cpu_mem_usage=True)
        print(f"  loaded via {how}: {type(model).__name__}")
        break
    except Exception as e:
        print(f"  {how} failed: {type(e).__name__}: {str(e)[:200]}")
if model is None:
    raise SystemExit("could not load; see errors above")

model.eval()
from perplexity import load_wikitext, perplexity
ids, nb = load_wikitext(tok)
dev = next(model.parameters()).device
met = perplexity(model, ids, 2048, device=dev, n_bytes=nb)
rec = {"tag": "gemma-3-27b-it-2bit-GuidedQuant-LNQ", "model": M,
       "quant": "2bit-guidedquant-lnq", "source": "published checkpoint", **met}
print("\n  PUBLISHED 2-BIT GEMMA-3-27B PERPLEXITY %.4f  BPB %s\n"
      % (met["perplexity"], met.get("bits_per_byte")))
open("/ephemeral/work/out/published.jsonl","a").write(json.dumps(rec)+"\n")
PY

echo "[$(date +%H:%M)] === PUBLISHED-CHECKPOINT EVAL DONE ==="
