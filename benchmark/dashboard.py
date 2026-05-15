"""Generates a fully self-contained HTML dashboard from benchmark results.

The output file has zero external dependencies — Chart.js is inlined from
benchmark/vendor/chart.min.js and all result data is embedded as JSON.
Recruiters can open dashboard.html directly in any browser, offline.

Called automatically by benchmark/runner.py after the benchmark completes.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

from benchmark.metrics import QueryMetrics

_VENDOR_DIR = Path(__file__).parent / "vendor"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
header { background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%); border-bottom: 1px solid #2d3748; padding: 28px 40px; }
header h1 { font-size: 1.6rem; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
header p { color: #718096; font-size: 0.85rem; margin-top: 4px; }
.badge { display: inline-block; background: #2d6a4f; color: #74c69d; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 99px; margin-left: 10px; vertical-align: middle; letter-spacing: 0.5px; }
main { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }
.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 36px; }
.card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 18px 20px; }
.card-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: #718096; margin-bottom: 6px; }
.card-value { font-size: 1.75rem; font-weight: 700; line-height: 1; }
.card-sub { font-size: 0.72rem; color: #718096; margin-top: 4px; }
.section-title { font-size: 1rem; font-weight: 600; color: #a0aec0; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #2d3748; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 36px; }
.chart-box { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }
.chart-box h3 { font-size: 0.82rem; font-weight: 600; color: #a0aec0; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px; }
canvas { max-height: 280px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 28px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; color: #a0aec0; }
.legend-dot { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
.query-block { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; margin-bottom: 20px; overflow: hidden; }
.query-header { padding: 16px 20px; background: #16213e; border-bottom: 1px solid #2d3748; display: flex; align-items: center; gap: 12px; }
.query-id { background: #2d6a4f; color: #74c69d; font-size: 0.7rem; font-weight: 700; padding: 3px 9px; border-radius: 99px; flex-shrink: 0; }
.query-text { font-size: 0.92rem; font-weight: 500; color: #e2e8f0; }
.strategy-tabs { display: flex; gap: 0; border-bottom: 1px solid #2d3748; overflow-x: auto; }
.tab-btn { background: none; border: none; color: #718096; font-size: 0.78rem; font-weight: 500; padding: 10px 16px; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; transition: all 0.15s; }
.tab-btn:hover { color: #a0aec0; }
.tab-btn.active { color: #63b3ed; border-bottom-color: #63b3ed; }
.strategy-panel { display: none; padding: 16px 20px; }
.strategy-panel.active { display: block; }
.expanded-query { background: #0f1117; border: 1px solid #2d3748; border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; color: #a0aec0; margin-bottom: 14px; }
.expanded-query strong { color: #718096; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 4px; }
.metrics-row { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.metric-pill { background: #0f1117; border: 1px solid #2d3748; border-radius: 8px; padding: 7px 12px; text-align: center; min-width: 80px; }
.metric-pill .m-label { font-size: 0.65rem; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-pill .m-value { font-size: 1rem; font-weight: 700; margin-top: 2px; }
.chunk-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.chunk-table th { text-align: left; padding: 7px 10px; color: #718096; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #2d3748; }
.chunk-table td { padding: 8px 10px; border-bottom: 1px solid #1e2535; vertical-align: top; }
.chunk-table tr:last-child td { border-bottom: none; }
.chunk-table tr:hover td { background: #0f1117; }
.rank-badge { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; font-size: 0.7rem; font-weight: 700; }
.rel-0 { background: #3d1515; color: #fc8181; }
.rel-1 { background: #3d3115; color: #f6c05e; }
.rel-2 { background: #1a3d2b; color: #68d391; }
.score-bar-wrap { display: flex; align-items: center; gap: 8px; }
.score-bar-bg { flex: 1; background: #0f1117; border-radius: 4px; height: 6px; overflow: hidden; }
.score-bar { height: 6px; border-radius: 4px; }
.score-text { font-size: 0.72rem; color: #718096; min-width: 44px; text-align: right; }
.drift-tag { font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 99px; }
.drift-low { background: #1a3d2b; color: #68d391; }
.drift-mid { background: #3d3115; color: #f6c05e; }
.drift-high { background: #3d1515; color: #fc8181; }
.adv-note { background: #1a1a2e; border: 1px solid #3d3d5c; border-radius: 8px; padding: 10px 14px; font-size: 0.75rem; color: #a0aec0; margin-top: 12px; }
.adv-note strong { color: #90cdf4; }
.gen-note { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; padding: 10px 16px; font-size: 0.74rem; color: #718096; margin-top: 32px; text-align: center; }
.gen-note code { color: #a0aec0; font-family: monospace; background: #0f1117; padding: 1px 5px; border-radius: 4px; }
@media (max-width: 900px) { .cards { grid-template-columns: repeat(3,1fr); } .chart-grid { grid-template-columns: 1fr; } }
@media (max-width: 600px) { .cards { grid-template-columns: 1fr 1fr; } header { padding: 20px; } main { padding: 20px 12px; } }
"""

# JavaScript that drives the dashboard — uses $DATA placeholder (no f-string
# needed so curly braces are literal and don't need escaping).
_JS_TEMPLATE = Template(r"""
const DATA = $DATA;

const COLOURS = {
  A:               { solid: '#63b3ed', alpha: 'rgba(99,179,237,0.75)'   },
  B_good_expansion: { solid: '#68d391', alpha: 'rgba(104,211,145,0.75)' },
  B_noisy_expansion:{ solid: '#fc8181', alpha: 'rgba(252,129,129,0.75)' },
  B_null_expansion: { solid: '#a0aec0', alpha: 'rgba(160,174,192,0.75)' },
  hybrid:           { solid: '#b794f4', alpha: 'rgba(183,148,244,0.75)' },
};

const STRATEGY_LABELS = {
  A:                'Strategy A (Raw)',
  B_good_expansion: 'B — Good Expansion',
  B_noisy_expansion:'B — Noisy Expansion',
  B_null_expansion: 'B — Null (passthrough)',
  hybrid:           'Hybrid (RRF)',
};

const STRATEGY_KEYS = Object.keys(COLOURS);
const QUERIES = DATA.map(q => q.query_id.toUpperCase());

function avg(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length; }

const agg = {};
STRATEGY_KEYS.forEach(sk => {
  const v = { ndcg:[], mrr:[], p3:[], latency:[], drift:[] };
  DATA.forEach(q => {
    const s = q.strategies.find(s=>s.name===sk);
    if (!s) return;
    v.ndcg.push(s.ndcg); v.mrr.push(s.mrr); v.p3.push(s.p3);
    v.latency.push(s.latency_ms);
    if (s.cosine_drift!==null) v.drift.push(s.cosine_drift);
  });
  agg[sk] = { ndcg:avg(v.ndcg), mrr:avg(v.mrr), p3:avg(v.p3), latency:avg(v.latency),
               drift: v.drift.length ? avg(v.drift) : null };
});

// ── Summary cards ────────────────────────────────────────────────────────────
const cardDefs = [
  { label:'Best Avg NDCG@3', key:'ndcg', fmt:v=>v.toFixed(3), colour:'#68d391' },
  { label:'Best Avg MRR@3',  key:'mrr',  fmt:v=>v.toFixed(3), colour:'#63b3ed' },
  { label:'Total Queries',   value:'5',  colour:'#b794f4' },
  { label:'Strategies Tested', value:'5', colour:'#f6c05e' },
  { label:'Fastest (ms)', key:'latency', fmt:v=>v.toFixed(1), lower:true, colour:'#fc8181' },
];
const cardsEl = document.getElementById('summary-cards');
cardDefs.forEach(def => {
  let val='', sub='';
  if (def.value) { val=def.value; }
  else {
    let best=null, bestK='';
    STRATEGY_KEYS.forEach(k => {
      const v=agg[k][def.key];
      if (best===null||(def.lower?v<best:v>best)){best=v;bestK=k;}
    });
    val=def.fmt(best); sub=STRATEGY_LABELS[bestK];
  }
  cardsEl.innerHTML+=`<div class="card"><div class="card-label">${def.label}</div>`
    +`<div class="card-value" style="color:${def.colour}">${val}</div>`
    +(sub?`<div class="card-sub">${sub}</div>`:'')+'</div>';
});

// ── Legend ───────────────────────────────────────────────────────────────────
const legendEl=document.getElementById('legend');
STRATEGY_KEYS.forEach(k=>{
  legendEl.innerHTML+=`<div class="legend-item"><div class="legend-dot" style="background:${COLOURS[k].solid}"></div>${STRATEGY_LABELS[k]}</div>`;
});

// ── Chart.js defaults ────────────────────────────────────────────────────────
Chart.defaults.color='#718096';
Chart.defaults.borderColor='#2d3748';
Chart.defaults.font.family='-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';

function makeGroupedBar(ctxId, metric) {
  new Chart(document.getElementById(ctxId),{
    type:'bar',
    data:{ labels:QUERIES, datasets:STRATEGY_KEYS.map(sk=>({
      label:STRATEGY_LABELS[sk],
      data:DATA.map(q=>{const s=q.strategies.find(s=>s.name===sk);return s?s[metric]:0;}),
      backgroundColor:COLOURS[sk].alpha, borderColor:COLOURS[sk].solid,
      borderWidth:1, borderRadius:4,
    }))},
    options:{ responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{ x:{grid:{color:'#1e2535'}}, y:{min:0,max:1.05,grid:{color:'#1e2535'},ticks:{stepSize:0.25}} }},
  });
}
makeGroupedBar('ndcgChart','ndcg');
makeGroupedBar('mrrChart','mrr');

new Chart(document.getElementById('latencyChart'),{
  type:'bar',
  data:{ labels:STRATEGY_KEYS.map(k=>STRATEGY_LABELS[k]),
    datasets:[{data:STRATEGY_KEYS.map(k=>+agg[k].latency.toFixed(1)),
      backgroundColor:STRATEGY_KEYS.map(k=>COLOURS[k].alpha),
      borderColor:STRATEGY_KEYS.map(k=>COLOURS[k].solid), borderWidth:1,borderRadius:4}]},
  options:{ indexAxis:'y',responsive:true,maintainAspectRatio:true,
    plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'#1e2535'}},y:{grid:{color:'#1e2535'}}} },
});

const driftKeys=['B_good_expansion','B_noisy_expansion','B_null_expansion'];
new Chart(document.getElementById('driftChart'),{
  type:'bar',
  data:{ labels:QUERIES, datasets:driftKeys.map(sk=>({
    label:STRATEGY_LABELS[sk],
    data:DATA.map(q=>{const s=q.strategies.find(s=>s.name===sk);return s&&s.cosine_drift!==null?s.cosine_drift:0;}),
    backgroundColor:COLOURS[sk].alpha, borderColor:COLOURS[sk].solid, borderWidth:1,borderRadius:4,
  }))},
  options:{ responsive:true,maintainAspectRatio:true,
    plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'#1e2535'}},y:{min:0,max:0.7,grid:{color:'#1e2535'},ticks:{stepSize:0.1}}} },
});

// ── Per-query blocks ─────────────────────────────────────────────────────────
function driftClass(v){ if(v===null)return''; if(v<0.25)return'drift-low'; if(v<0.45)return'drift-mid'; return'drift-high'; }
function relLabel(r){ return['Irrelevant','Partial','Relevant'][r]; }

const qBlocksEl=document.getElementById('query-blocks');
DATA.forEach((q,qi)=>{
  const bid=`q-${qi}`;
  const tabs=q.strategies.map((s,si)=>
    `<button class="tab-btn ${si===0?'active':''}" onclick="switchTab('${bid}',${si})" id="${bid}-tab-${si}">${STRATEGY_LABELS[s.name]}</button>`
  ).join('');

  const panels=q.strategies.map((s,si)=>{
    const expanded = s.expanded_query_text && s.expanded_query_text!==q.query_text
      ? `<div class="expanded-query"><strong>Expanded Query</strong>${s.expanded_query_text}</div>` : '';

    const metrics=`<div class="metrics-row">
      <div class="metric-pill"><div class="m-label">MRR@3</div><div class="m-value" style="color:${s.mrr===1?'#68d391':s.mrr>0?'#f6c05e':'#fc8181'}">${s.mrr.toFixed(4)}</div></div>
      <div class="metric-pill"><div class="m-label">NDCG@3</div><div class="m-value" style="color:${s.ndcg>=0.9?'#68d391':s.ndcg>=0.5?'#f6c05e':'#fc8181'}">${s.ndcg.toFixed(4)}</div></div>
      <div class="metric-pill"><div class="m-label">P@3</div><div class="m-value" style="color:#a0aec0">${s.p3.toFixed(4)}</div></div>
      <div class="metric-pill"><div class="m-label">Latency</div><div class="m-value" style="color:#63b3ed">${s.latency_ms.toFixed(1)}<span style="font-size:0.6rem;color:#718096"> ms</span></div></div>
      ${s.cosine_drift!==null?`<div class="metric-pill"><div class="m-label">Cosine Drift</div><div class="m-value ${driftClass(s.cosine_drift)}" style="font-size:0.9rem">${s.cosine_drift.toFixed(4)}</div></div>`:''}
    </div>`;

    const rows=s.chunks.map(c=>{
      const barW=Math.round(c.score*100);
      return `<tr>
        <td><span class="rank-badge rel-${c.relevance}">${c.rank}</span></td>
        <td style="font-family:monospace;font-size:0.78rem;color:#a0aec0">${c.id}</td>
        <td><div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar" style="width:${barW}%;background:${COLOURS[s.name].solid}"></div></div><span class="score-text">${c.score.toFixed(4)}</span></div></td>
        <td><span class="drift-tag ${c.relevance===2?'drift-low':c.relevance===1?'drift-mid':'drift-high'}">${relLabel(c.relevance)}</span></td>
      </tr>`;
    }).join('');

    const advNote = q.query_id==='q4'
      ? `<div class="adv-note"><strong>Adversarial Query:</strong> No CAP-theorem content exists in the GCP corpus. Every strategy returns MRR=0 and NDCG=0 — correct behaviour. The system surfaces low-confidence results rather than fabricating an answer.</div>`
      : '';

    return `<div class="strategy-panel ${si===0?'active':''}" id="${bid}-panel-${si}">${expanded}${metrics}<table class="chunk-table"><thead><tr><th>Rank</th><th>Chunk ID</th><th>Cosine Score</th><th>Relevance</th></tr></thead><tbody>${rows}</tbody></table>${advNote}</div>`;
  }).join('');

  qBlocksEl.innerHTML+=`<div class="query-block" id="${bid}"><div class="query-header"><span class="query-id">${q.query_id.toUpperCase()}</span><span class="query-text">${q.query_text}</span></div><div class="strategy-tabs">${tabs}</div>${panels}</div>`;
});

function switchTab(bid,idx){
  document.querySelectorAll(`#${bid} .tab-btn`).forEach((b,i)=>b.classList.toggle('active',i===idx));
  document.querySelectorAll(`#${bid} .strategy-panel`).forEach((p,i)=>p.classList.toggle('active',i===idx));
}
""")

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Benchmark Dashboard</title>
<style>$CSS</style>
</head>
<body>

<header>
  <h1>RAG Benchmark Dashboard <span class="badge">LIVE RESULTS</span></h1>
  <p>Strategy A (Raw Vector Search) vs Strategy B (AI-Enhanced Retrieval) vs Hybrid &mdash;
     sentence-transformers mpnet-768 &middot; FAISS IndexFlatIP &middot; Mocked Vertex AI SDK</p>
</header>

<main>
  <div class="cards" id="summary-cards"></div>
  <p class="section-title">Aggregate Metrics by Strategy</p>
  <div class="chart-grid">
    <div class="chart-box"><h3>NDCG@3 per Query &amp; Strategy</h3><canvas id="ndcgChart"></canvas></div>
    <div class="chart-box"><h3>MRR@3 per Query &amp; Strategy</h3><canvas id="mrrChart"></canvas></div>
    <div class="chart-box"><h3>Avg Latency (ms) by Strategy</h3><canvas id="latencyChart"></canvas></div>
    <div class="chart-box"><h3>Cosine Drift (Strategy B variants)</h3><canvas id="driftChart"></canvas></div>
  </div>
  <div class="legend" id="legend"></div>
  <p class="section-title">Per-Query Retrieval Breakdown</p>
  <div id="query-blocks"></div>
  <div class="gen-note">
    Auto-generated by <code>python -m benchmark.runner</code> &mdash;
    open <code>dashboard.html</code> in any browser, no server required.
  </div>
</main>

<script>$CHARTJS</script>
<script>$JS</script>
</body>
</html>
""")


def write_dashboard(path: Path, results: list[QueryMetrics]) -> None:
    """Render a fully self-contained HTML dashboard and write it to *path*.

    The file has zero external dependencies: Chart.js is inlined from
    benchmark/vendor/chart.min.js and all result data is embedded as JSON.

    Args:
        path: Destination file path (e.g. Path("dashboard.html")).
        results: List of QueryMetrics from engine.benchmark(), optionally
                 annotated with expanded_query_text by runner.py.
    """
    chartjs = (_VENDOR_DIR / "chart.min.js").read_text(encoding="utf-8")
    json_data = _results_to_json(results)
    js = _JS_TEMPLATE.safe_substitute(DATA=json_data)
    html = _HTML_TEMPLATE.safe_substitute(CSS=_CSS, CHARTJS=chartjs, JS=js)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _results_to_json(results: list[QueryMetrics]) -> str:
    """Convert QueryMetrics list to the JSON shape expected by the dashboard JS."""
    by_query: dict[str, list[QueryMetrics]] = {}
    for r in results:
        by_query.setdefault(r.query_id, []).append(r)

    output = []
    for qid, qresults in sorted(by_query.items()):
        strategies = []
        for qm in qresults:
            strategies.append({
                "name": qm.strategy,
                "expansion_mode": qm.expansion_mode,
                "expanded_query_text": getattr(qm, "expanded_query_text", None),
                "chunks": [
                    {
                        "id": rc.chunk_id,
                        "score": round(rc.score, 6),
                        "relevance": rc.relevance,
                        "rank": rc.rank,
                    }
                    for rc in qm.retrieved
                ],
                "mrr": round(qm.mrr, 4),
                "ndcg": round(qm.ndcg, 4),
                "p3": round(qm.p3, 4),
                "latency_ms": round(qm.latency_ms, 3),
                "cosine_drift": round(qm.cosine_drift, 4) if qm.cosine_drift is not None else None,
            })
        output.append({
            "query_id": qid,
            "query_text": qresults[0].query_text,
            "strategies": strategies,
        })

    return json.dumps(output, indent=2, ensure_ascii=False)
