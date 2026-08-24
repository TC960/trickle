"""Experiment registry: provenance for every result, so nothing is unaccounted.

Each record captures what was run, on what code, with which library versions, on
which machine -- so a number can always be traced back and reproduced. Records
are append-only JSONL; nothing is ever overwritten.

Two fields make A/B structure explicit rather than implied:

    arm      "control" or "treatment"
    pair_id  ties a treatment to the control it must be compared against

A treatment without a matching control in the same pair is flagged by report.py
rather than silently reported, because a compression number with no baseline is
not a result.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
import uuid
from pathlib import Path

REGISTRY = Path("/ephemeral/work/out/registry.jsonl")


def _code_fingerprint():
    """SHA256 over the experiment sources, so results pin to a code state."""
    code = Path(__file__).parent
    h = hashlib.sha256()
    for name in sorted(p.name for p in code.glob("*.py")):
        h.update(name.encode())
        h.update((code / name).read_bytes())
    return h.hexdigest()[:16]


def _env():
    info = {"host": socket.gethostname(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def record(*, tag, kind, model, arm, pair_id, params, results, notes=None):
    """Append one fully-provenanced result.

    kind     what family of experiment ("perplexity", "embed", "distill", "bench")
    arm      "control" | "treatment"
    pair_id  shared key linking a treatment to its control
    """
    if arm not in ("control", "treatment"):
        raise ValueError(f"arm must be control|treatment, got {arm!r}")

    rec = {
        "run_id": uuid.uuid4().hex[:12],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": tag, "kind": kind, "model": model,
        "arm": arm, "pair_id": pair_id,
        "params": params, "results": results,
        "code_sha": _code_fingerprint(),
        "env": _env(),
        "notes": notes,
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a") as h:
        h.write(json.dumps(rec) + "\n")
    print(f"  [registry] {rec['run_id']} {tag} ({arm}, pair={pair_id})", flush=True)
    return rec


def load():
    if not REGISTRY.exists():
        return []
    out = []
    for line in REGISTRY.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out
