"""Live status dashboard. Serves a self-refreshing page on :8777.

    python dashboard.py            # then port-forward 8777 to your laptop

Reads whatever exists on disk -- result JSONLs, tmux sessions, nvidia-smi, log
tails -- so it never needs to be told what's running. Empty states are normal.
"""

import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORK = Path("/ephemeral/work")
OUT, LOGS = WORK / "out", WORK / "logs"
PORT = 8777
STARTED = time.time()


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def gpus():
    raw = sh("nvidia-smi --query-gpu=index,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu --format=csv,noheader,nounits")
    out = []
    for line in raw.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            out.append({"idx": int(p[0]), "used_gb": round(int(p[1]) / 1024, 1),
                        "total_gb": round(int(p[2]) / 1024, 1),
                        "util": int(p[3]), "temp": int(p[4])})
    return out


def tail(name, n=14):
    f = LOGS / name
    if not f.exists():
        return ""
    lines = [l for l in f.read_text(errors="replace").splitlines()
             if l.strip() and "it/s]" not in l and "examples/s]" not in l]
    return "\n".join(lines[-n:])


def current_step(log):
    """Most recent '##### model :: step' banner, i.e. what's running now."""
    text = tail(log, 400)
    hits = re.findall(r"#+ (\S+) :: (\S+) :: (\d\d:\d\d:\d\d)", text)
    if not hits:
        return None
    model, step, when = hits[-1]
    done = f"exit=" in text.split(f":: {step} ::")[-1]
    return {"model": model, "step": step, "since": when, "finished": done}


def active_runs():
    """What is executing right now, from the logs of in-flight runs.

    Counting result-file lines is wrong: those accumulate across every run ever
    performed, which is how the dashboard came to claim "120 of 60 blocks".
    """
    out = []
    for f in sorted(LOGS.glob("seq_d*.log")) + sorted(LOGS.glob("seq_qwen*.log")):
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        trained = len(re.findall(r"block +\d+: naive", text))
        kept = len(re.findall(r"block \d+: kept bf16", text))
        finished = "PERPLEXITY" in text
        target = trained + kept
        out.append({"run": f.stem, "trained": trained, "kept_bf16": kept,
                    "total_blocks": target or None, "finished": finished})
    return out


def status():
    ppl = read_jsonl(OUT / "ppl.jsonl")
    embed = read_jsonl(OUT / "embed.jsonl")
    distill = read_jsonl(OUT / "distill.jsonl")
    depth = read_jsonl(OUT / "seq_depth.jsonl")
    trim = read_jsonl(OUT / "vocab_trim.jsonl")
    vocab = read_jsonl(OUT / "vocab.jsonl")

    # Baselines per model, so every row can show a delta.
    base = {r["model"]: r["perplexity"] for r in ppl if r.get("quant") == "none"}
    for r in ppl + embed:
        b = base.get(r.get("model"))
        r["delta_pct"] = round((r["perplexity"] / b - 1) * 100, 3) if b else None

    sessions = [s.split(":")[0] for s in sh("tmux ls 2>/dev/null").splitlines()]
    return {
        "now": time.strftime("%H:%M:%S"),
        "uptime_min": round((time.time() - STARTED) / 60, 1),
        "gpus": gpus(),
        "sessions": sessions,
        "disk": sh("df -h /ephemeral | tail -1 | awk '{print $3\" / \"$2}'"),
        "ppl": ppl, "embed": embed, "distill": distill,
        "depth": depth, "trim": trim, "vocab": vocab,
        "steps": {n: current_step(f"sweep_{n}.log") for n in ("qwen", "gemma")},
        "logs": {n: tail(f"sweep_{n}.log", 12) for n in ("qwen", "gemma")},
        "depth_log": tail("depth.log", 14),
        "trim_log": tail("trim.log", 14),
        "distill_log": tail("distill.log", 12),
        "docs": {n: (OUT / f"{n}.md").read_text()[:60000]
                 for n in ("AUDIT", "REPORT", "BACKLOG")
                 if (OUT / f"{n}.md").exists()},
        "counts": {"ppl": len(ppl), "embed": len(embed),
                   "distill_records": len(distill), "depth": len(depth),
                   "trim": len(trim)},
        "active": active_runs(),
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ternary compression - live</title>
<style>
/* Dark surface + validated categorical slots 1 (blue) and 2 (orange).
   Palette verified with the dataviz validator: adjacent CVD dE 26.8,
   normal-vision 31.8, both clear of the floors on this surface. */
:root{
  --plane:#0d0d0d; --surface:#1a1a19;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:17px;font-weight:600;margin:0 0 2px}
.sub{color:var(--muted);font-size:12px;margin-bottom:20px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
  padding:16px 18px;margin-bottom:14px}
.card h2{font-size:12px;font-weight:600;color:var(--ink2);margin:0 0 4px;
  letter-spacing:.06em;text-transform:uppercase}
.note{font-size:12px;color:var(--muted);line-height:1.55;margin:0 0 13px;max-width:70ch}
.note b{color:var(--ink2);font-weight:500}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:26px;font-weight:600;margin-top:6px;line-height:1.1}
.tile .n{font-size:11px;color:var(--ink2);margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-weight:500;color:var(--muted);font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;padding:6px 10px 8px;
  border-bottom:1px solid var(--axis)}
td{padding:7px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.num{text-align:right}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;
  padding:2px 8px;border-radius:999px;border:1px solid var(--ring)}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.tab{background:#1f1f1e;color:var(--ink2);border:1px solid var(--ring);
  border-radius:6px;padding:5px 13px;font-size:12px;cursor:pointer;
  font-family:inherit}
.tab:hover{background:#2a2a28}
.tab.on{background:var(--s1);color:#fff;border-color:var(--s1)}
.ev{display:grid;grid-template-columns:auto 1fr;gap:7px 12px;font-size:13px;
  align-items:baseline}
.ev .st{font-size:11px;padding:1px 8px;border-radius:999px;white-space:nowrap;
  border:1px solid var(--ring)}
pre{background:#111110;border:1px solid var(--grid);border-radius:8px;
  padding:11px 13px;font-size:11.5px;line-height:1.55;overflow-x:auto;
  color:var(--ink2);margin:0;max-height:230px;white-space:pre-wrap}
.legend{display:flex;gap:16px;font-size:11.5px;color:var(--ink2);margin-bottom:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;flex:none}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
.empty{color:var(--muted);font-size:12.5px;padding:8px 0}
#tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--ring);
  border-radius:6px;padding:6px 9px;font-size:11.5px;opacity:0;transition:opacity .1s;
  z-index:9;white-space:nowrap}
</style></head><body>
<div class="wrap">
  <h1>ternary compression - live</h1>
  <div class="sub" id="sub">connecting...</div>
  <div class="tiles" id="tiles"></div>

  <div class="card" style="border-color:rgba(250,178,25,.35)">
    <h2>evidence status - read this before trusting any number</h2>
    <p class="note">Nine bugs were found in this project's own code, and
    <b>six of them produced believable but wrong results</b> rather than
    crashing. Numbers below are tagged by how much they are actually worth.</p>
    <div id="evidence"></div></div>

  <div class="card"><h2>documents</h2>
    <p class="note">AUDIT = every claim and its evidence quality.
    REPORT = auto-generated results. BACKLOG = methodology log and next steps.</p>
    <div style="display:flex;gap:8px;margin-bottom:10px">
      <button class="tab" data-doc="AUDIT">AUDIT</button>
      <button class="tab" data-doc="REPORT">REPORT</button>
      <button class="tab" data-doc="BACKLOG">BACKLOG</button>
    </div>
    <pre id="doc" style="max-height:520px"></pre></div>
  <div class="card"><h2>perplexity by method</h2>
    <p class="note">How much quality each compression method costs. <b>Perplexity =
    how surprised the model is by real text; lower is better.</b> Only compare a
    row to its OWN model's baseline - different models use different tokenizers,
    so their absolute numbers are not comparable. <b>Negative delta means it got
    better than baseline.</b></p><div id="ppl"></div></div>
  <div class="card"><h2>embedding compression ablation</h2>
    <p class="note">The embedding table (one vector per vocabulary token) turned
    out to be the real memory bottleneck - on small models it is bigger than all
    the quantized transformer layers combined. This measures what shrinking it
    costs. <b>int8</b> = 8 bits/weight. <b>int4-g32</b> = 4 bits with one scale
    per 32 weights (finer = better). <b>svd-rN</b> = low-rank: rebuild the table
    from N "concept directions" instead of all of them. <b>untied</b> = give the
    output head its own full-precision copy instead of sharing.</p>
    <div id="embed"></div></div>
  <div class="card"><h2>per-block distillation fidelity</h2>
    <p class="note">Converting to ternary (every weight becomes just -1, 0 or +1)
    layer by layer. <b>Blue = naive rounding</b>, which is what you get for free.
    <b>Orange = after training that layer to imitate the original.</b> 1.0 means
    the compressed layer reproduces the original's output exactly. Note this is
    per-layer fidelity, NOT overall model quality - small errors compound across
    60 layers, so the end-to-end perplexity above is the number that counts.</p>
    <div class="legend">
      <span><i class="sw" style="background:var(--s1)"></i>naive PTQ</span>
      <span><i class="sw" style="background:var(--s2)"></i>after distillation</span>
    </div>
    <div id="chart"></div></div>
  <div class="card"><h2>ternary depth sweep</h2>
    <p class="note">How many of the 60 decoder blocks can go ternary before the
    model breaks. Each row is an independent run against the same bf16 baseline
    (perplexity 5.1876). <b>Errors compound down the layer stack</b>, so this is
    a curve, not a pass/fail - the useful answer is likely "ternarize the middle
    N layers, keep the rest higher-precision".</p>
    <div id="depth"></div></div>

  <div class="card"><h2>vocabulary trimming</h2>
    <p class="note">The output projection needs every vocabulary row on every
    token, so at bf16 it costs ~2.6 GB of streaming bandwidth per token - about
    the same as 20 ternary layers. Trimming unused tokens is the lever.
    <b>merge_closure</b> = extra tokens that must be kept so the BPE merge table
    stays valid; without them the tokenizer silently breaks. Cost: text becomes
    10-15% more tokens, which is a throughput hit.</p>
    <div id="trim"></div></div>

  <div class="grid2">
    <div class="card"><h2>gemma sweep</h2>
      <p class="note">Raw log for google/gemma-4-31B (60 layers, tied
      embeddings).</p><pre id="lg"></pre></div>
    <div class="card"><h2>qwen sweep</h2>
      <p class="note">Raw log for Qwen/Qwen3.8-27B (64 layers, 48 of them linear
      attention, untied embeddings).</p><pre id="lq"></pre></div>
  </div>
  <div class="grid2">
    <div class="card"><h2>depth sweep log</h2><pre id="ldepth"></pre></div>
    <div class="card"><h2>vocab trim log</h2><pre id="ltrim"></pre></div>
  </div>
  <div class="card"><h2>distillation log</h2>
    <p class="note">Live ternary conversion. Each block trains for 200 steps
    (~15s) to match its full-precision teacher's output.</p>
    <pre id="ld"></pre></div>
</div>
<div id="tip"></div>
<script>
const tip=document.getElementById('tip');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+12,innerWidth-190)+'px';tip.style.top=(e.clientY-34)+'px';}
function hideTip(){tip.style.opacity=0;}

function statusPill(s){
  if(!s) return '<span class="pill"><i class="dot" style="background:var(--muted)"></i>idle</span>';
  const c=s.finished?'var(--good)':'var(--warn)', t=s.finished?'done':'running';
  return `<span class="pill"><i class="dot" style="background:${c}"></i>${t} &middot; ${esc(s.step)}</span>`;
}

function tiles(d){
  // The run still in flight, if any -- this is live progress, not a total.
  const act=(d.active||[]).filter(a=>!a.finished).slice(-1)[0]
            || (d.active||[]).slice(-1)[0] || null;
  const g=d.gpus.map(g=>`gpu${g.idx} ${g.used_gb}/${g.total_gb}GB &middot; ${g.util}% &middot; ${g.temp}&deg;C`).join('<br>');
  return [
    ['results logged',`${d.counts.ppl+d.counts.embed+d.counts.depth}`,
      `${d.counts.ppl} ppl / ${d.counts.embed} embed / ${d.counts.depth} depth`],
    ['active run',act?`${act.trained}/${act.total_blocks||'?'}`:'idle',
      act?`${act.run} - ${act.trained} ternary, ${act.kept_bf16} kept bf16`
         :'no distillation running'],
    ['gpu busiest',(d.gpus.length?Math.max(...d.gpus.map(x=>x.util)):0)+'%',
      (g||'no gpu data')+'<br><i>one card idling is normal: the model is split across both, so they alternate</i>'],
    ['disk used',(d.disk||'-'),'on /ephemeral'],
    ['dashboard up',d.uptime_min+'m','last poll '+d.now],
  ].map(([k,v,n])=>`<div class="tile"><div class="k">${k}</div>
     <div class="v">${v}</div><div class="n">${n}</div></div>`).join('');
}

function pplTable(rows){
  if(!rows.length) return '<div class="empty">no results yet</div>';
  rows=rows.slice().sort((a,b)=>(a.model||'').localeCompare(b.model)||a.tag.localeCompare(b.tag));
  return `<table><thead><tr><th>run</th><th class="num">perplexity</th>
    <th class="num">delta</th><th class="num">footprint</th><th class="num">eval</th></tr></thead><tbody>`
    + rows.map(r=>{
        const d=r.delta_pct, sign=d>0?'+':'';
        // Delta text is neutral ink unless it is the baseline itself; a color
        // here would imply a series identity it does not have.
        const dc = d===null||d===undefined ? 'var(--muted)' : (Math.abs(d)<1?'var(--ink2)':'var(--warn)');
        return `<tr><td>${esc(r.tag)}</td>
          <td class="num">${r.perplexity.toFixed(4)}</td>
          <td class="num" style="color:${dc}">${d===null||d===undefined?'baseline':sign+d+'%'}</td>
          <td class="num">${r.footprint_gb?r.footprint_gb+' GB':r.net_mb?r.net_mb+' MB':'-'}</td>
          <td class="num">${r.eval_seconds?Math.round(r.eval_seconds)+'s':'-'}</td></tr>`;
      }).join('')+'</tbody></table>';
}

function chart(rows){
  const el=document.getElementById('chart');
  if(!rows.length){el.innerHTML='<div class="empty">distillation has not started</div>';return;}
  rows=rows.slice().sort((a,b)=>a.block-b.block);
  const W=1080,H=250,P={t:14,r:16,b:30,l:44};
  const xs=i=>P.l+(rows.length<2?0:i*(W-P.l-P.r)/(rows.length-1));
  const all=rows.flatMap(r=>[r.naive_cosine,r.trained_cosine]).filter(v=>v!=null);
  let lo=Math.min(...all), hi=Math.max(...all);
  lo=Math.max(0,lo-(hi-lo)*.15||lo-.02); hi=Math.min(1,hi+(hi-lo)*.15||hi+.02);
  const ys=v=>P.t+(1-(v-lo)/(hi-lo||1))*(H-P.t-P.b);
  const path=k=>rows.map((r,i)=>(i?'L':'M')+xs(i)+' '+ys(r[k])).join(' ');
  let g='';
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=ys(v);
    g+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>
        <text x="${P.l-8}" y="${y+3.5}" fill="var(--muted)" font-size="10" text-anchor="end">${v.toFixed(3)}</text>`;}
  // 2px lines, >=8px markers, 2px surface ring on marks so overlaps stay readable.
  let m='';
  rows.forEach((r,i)=>{
    [['naive_cosine','var(--s1)','naive PTQ'],['trained_cosine','var(--s2)','distilled']].forEach(([k,c,lab])=>{
      if(r[k]==null)return;
      m+=`<circle cx="${xs(i)}" cy="${ys(r[k])}" r="4.5" fill="${c}" stroke="var(--surface)" stroke-width="2"
          data-t="block ${r.block} &middot; ${lab}: <b>${r[k].toFixed(4)}</b>"/>`;});
  });
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px">
    ${g}<line x1="${P.l}" y1="${H-P.b}" x2="${W-P.r}" y2="${H-P.b}" stroke="var(--axis)" stroke-width="1"/>
    <path d="${path('naive_cosine')}" fill="none" stroke="var(--s1)" stroke-width="2"/>
    <path d="${path('trained_cosine')}" fill="none" stroke="var(--s2)" stroke-width="2"/>
    ${m}
    <text x="${P.l}" y="${H-9}" fill="var(--muted)" font-size="10">block 0</text>
    <text x="${W-P.r}" y="${H-9}" fill="var(--muted)" font-size="10" text-anchor="end">block ${rows[rows.length-1].block}</text>
  </svg>`;
  el.querySelectorAll('circle').forEach(c=>{
    c.addEventListener('mousemove',e=>showTip(e,c.dataset.t));
    c.addEventListener('mouseleave',hideTip);});
}

function depthTable(rows, base){
  if(!rows.length) return '<div class="empty">depth sweep has not produced results yet</div>';
  rows=rows.slice().sort((a,b)=>(b.blocks_ternary||0)-(a.blocks_ternary||0));
  return `<table><thead><tr><th>run</th><th class="num">blocks ternary</th>
    <th class="num">perplexity</th><th class="num">delta</th>
    <th class="num">bits/byte</th></tr></thead><tbody>`
    + rows.map(r=>{
        const n=r.blocks_ternary!=null?r.blocks_ternary:(r.max_blocks||'-');
        const d=base?((r.perplexity/base-1)*100):null;
        const dtxt=d===null?'-':(Math.abs(d)>1000?d.toExponential(2)+'%':d.toFixed(2)+'%');
        const dc=d===null?'var(--muted)':(d<5?'var(--good)':(d<50?'var(--warn)':'var(--crit)'));
        return `<tr><td>${esc(r.tag||'-')}</td><td class="num">${n}</td>
          <td class="num">${r.perplexity!=null?r.perplexity.toFixed(4):'-'}</td>
          <td class="num" style="color:${dc}">${dtxt}</td>
          <td class="num">${r.bits_per_byte!=null?r.bits_per_byte:'-'}</td></tr>`;
      }).join('')+'</tbody></table>';
}

function trimTable(rows){
  if(!rows.length) return '<div class="empty">no trimming runs yet</div>';
  return `<table><thead><tr><th>model</th><th class="num">vocab</th>
    <th class="num">table</th><th class="num">saved</th>
    <th class="num">shrink</th><th class="num">token inflation</th>
    <th>valid</th></tr></thead><tbody>`
    + rows.slice().sort((a,b)=>a.vocab_after-b.vocab_after).map(r=>{
        const infl=(r.probe_tokens&&r.probe_tokens_original)
          ? '+'+(((r.probe_tokens/r.probe_tokens_original)-1)*100).toFixed(1)+'%' : '-';
        const ok=r.roundtrip_ok;
        return `<tr><td>${esc((r.model||'').split('/').pop())}</td>
          <td class="num">${r.vocab_before} &rarr; ${r.vocab_after}</td>
          <td class="num">${r.table_mb_after} MB</td>
          <td class="num">${r.saved_mb} MB</td>
          <td class="num">${r.shrink_x}x</td>
          <td class="num">${infl}</td>
          <td><span class="pill"><i class="dot" style="background:${ok?'var(--good)':'var(--crit)'}"></i>${ok?'ok':'broken'}</span></td></tr>`;
      }).join('')+'</tbody></table>';
}

const EVIDENCE = [
  ['solid','streaming engine bit-exact vs reference (max|d|=0, all 30 layers)'],
  ['solid','int8 +0.49% perplexity AND 3.8% flip rate - both measured'],
  ['solid','nf4 costs Gemma 5.6x more than Qwen (+5.03% vs +0.90%)'],
  ['solid','~80% of vocabulary never fires; but that is only 3.6% of params'],
  ['solid','MLP = 67.5% of params (computed from the checkpoint)'],
  ['solid','frequency-damage DIRECTION depends on the method'],
  ['partial','bit-width curve is PERPLEXITY ONLY - no flip rate, no GSM8K yet'],
  ['partial','"4-bit beats nf4" is a 1.9% gap, single seed, no variance estimate'],
  ['partial','depth knee used FIRST-N layers, conflating position with tolerance'],
  ['partial','sensitivity profile ran on a truncated eval for speed'],
  ['partial','embedding "untied" runs invalid - was_tied bug made --untie a no-op'],
  ['missing','streaming engine has NEVER run on Gemma 4 - only BitNet-2B'],
  ['missing','no inference speed (tok/s) measured for any Gemma config'],
  ['missing','GPTQ implemented but never beaten round-to-nearest on real data'],
  ['missing','mixed precision never ran; learnable-threshold never completed'],
  ['missing','vocab trimming has size numbers but zero quality measurements'],
  ['missing','no published baseline reproduced by us (GuidedQuant eval OOMd)'],
  ['missing','single seed everywhere - no error bars on anything'],
];
function evidence(){
  const col={solid:'var(--good)',partial:'var(--warn)',missing:'var(--crit)'};
  const lbl={solid:'solid',partial:'partial',missing:'NOT DONE'};
  return EVIDENCE.map(([k,t])=>
    `<span class="st" style="color:${col[k]};border-color:${col[k]}">${lbl[k]}</span>
     <span style="color:var(--ink2)">${esc(t)}</span>`).join('');
}
let DOCS={}, curDoc='AUDIT';
function showDoc(){
  document.querySelectorAll('.tab').forEach(b=>
    b.classList.toggle('on', b.dataset.doc===curDoc));
  document.getElementById('doc').textContent =
    DOCS[curDoc] || '(not generated yet)';
}
document.addEventListener('click', e=>{
  if(e.target.classList.contains('tab')){ curDoc=e.target.dataset.doc; showDoc(); }
});

async function poll(){
  try{
    const d=await (await fetch('/api/status')).json();
    document.getElementById('sub').innerHTML =
      `${d.now} &middot; sessions: ${d.sessions.join(', ')||'none'} &middot; `
      + `gemma ${statusPill(d.steps.gemma)} qwen ${statusPill(d.steps.qwen)}`;
    document.getElementById('tiles').innerHTML=tiles(d);
    document.getElementById('evidence').innerHTML=evidence();
    DOCS=d.docs||{}; showDoc();
    document.getElementById('ppl').innerHTML=pplTable(d.ppl);
    document.getElementById('embed').innerHTML=pplTable(d.embed);
    chart(d.distill);
    const gbase=(d.ppl.find(r=>r.quant==='none'&&(r.model||'').includes('gemma'))||{}).perplexity;
    document.getElementById('depth').innerHTML=depthTable(
      d.depth.concat(d.ppl.filter(r=>(r.quant||'').startsWith('ternary'))), gbase);
    document.getElementById('trim').innerHTML=trimTable(d.trim);
    document.getElementById('ldepth').textContent=d.depth_log||'(nothing yet)';
    document.getElementById('ltrim').textContent=d.trim_log||'(nothing yet)';
    document.getElementById('lg').textContent=d.logs.gemma||'(nothing yet)';
    document.getElementById('lq').textContent=d.logs.qwen||'(nothing yet)';
    document.getElementById('ld').textContent=d.distill_log||'(not started)';
  }catch(e){document.getElementById('sub').textContent='lost connection to dashboard: '+e;}
}
poll(); setInterval(poll,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status()).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"dashboard on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
