# RAG Benchmark — Strategy A vs Strategy B vs Hybrid

A complete retrieval benchmarking system comparing raw vector search, query expansion,
and RRF hybrid fusion across a GCP-domain corpus, with a Vertex AI migration path.

---

## Quickstart — Pick One Method

### Method 1: Windows (one double-click)

```
Double-click run.bat
```

Or from Command Prompt:

```cmd
run.bat
```

To run tests instead:

```cmd
run.bat --test
```

---

### Method 2: Linux / macOS (one command)

```bash
chmod +x run.sh && ./run.sh
```

To run tests instead:

```bash
./run.sh --test
```

---

### Method 3: Docker (runs on any OS with Docker installed)

```bash
# Build image (downloads embedding model during build, ~1.4 GB)
docker compose build

# Run benchmark — output files appear in the current directory
docker compose run --rm rag-benchmark

# Run tests
docker compose run --rm rag-test
```

---

### Method 4: Manual steps (Python 3.10+)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the benchmark
python -m benchmark.runner

# 4. Run tests
pytest tests/ -v
```

---

### Method 5: pip install (developer / editable)

```bash
pip install -e ".[dev]"   # installs package + dev dependencies
rag-benchmark             # CLI entry point
pytest tests/ -v
```

---

## First-Run Note

On the **very first run** the embedding model
(`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) is downloaded
from HuggingFace — approximately **1.4 GB**.
Subsequent runs use the cached model and take only a few seconds.

If you are in a restricted network environment, set `HF_HOME` to a directory
that already contains the model, or use the Docker method which pre-bakes the
model into the image.

---

## Viewing Results — Interactive Dashboard

The repo ships a pre-built **`dashboard.html`** that you can open right now:

```
# macOS
open dashboard.html

# Windows
start dashboard.html

# Linux
xdg-open dashboard.html
```

No server, no Python, no internet required — Chart.js is bundled inside the file.

The dashboard shows:
- **Summary cards** — best avg NDCG@3, MRR@3, total queries, fastest strategy
- **4 charts** — NDCG@3 and MRR@3 per query × strategy, avg latency, cosine drift
- **Per-query breakdown** — click any strategy tab to compare chunk rankings, cosine scores, and relevance labels side-by-side

---

## What the Benchmark Produces

After `python -m benchmark.runner` completes, three files appear in the working directory:

| File | Description |
|------|-------------|
| `benchmark_results.json` | Machine-readable metrics for all 5 queries × 5 strategies |
| `retrieval_benchmark.md` | Human-readable comparison report with expanded query text |
| `dashboard.html` | Self-contained interactive dashboard (no internet needed) |

All three files are committed to the repo for reference. Running the benchmark
overwrites them with fresh results.

---

## Prerequisites

| Tool | Minimum version | How to check |
|------|----------------|--------------|
| Python | 3.10 | `python --version` |
| pip | 21.0 | `pip --version` |
| Docker (optional) | 20.10 | `docker --version` |

No GCP credentials or API keys are required — all Vertex AI calls are mocked.

---

## Project Structure

```
rag-benchmark/
├── run.bat                  # Windows one-click runner
├── run.sh                   # Linux/macOS one-click runner
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
├── Makefile                 # Developer shortcuts (make run / make test)
├── config.yaml              # All runtime parameters
├── requirements.txt         # Pinned dependencies
├── pyproject.toml           # Installable package + CLI entry point
│
├── core/
│   ├── config.py            # Pydantic settings (loaded from config.yaml)
│   └── rag_engine.py        # RAGEngine: ingest → benchmark → export_report
│
├── embeddings/
│   ├── base.py              # EmbeddingModel Protocol
│   ├── local.py             # gecko simulation (768-dim multilingual mpnet)
│   └── sparse.py            # BM25 sparse index
│
├── vector_store/
│   ├── base.py              # VectorStore Protocol
│   ├── faiss_store.py       # FAISS IndexFlatIP + L2 normalisation
│   └── numpy_store.py       # Pure NumPy fallback
│
├── retrieval/
│   ├── strategy_a.py        # RawVectorRetriever (Strategy A)
│   ├── strategy_b.py        # ExpandedQueryRetriever (Strategy B)
│   └── hybrid.py            # Reciprocal Rank Fusion
│
├── expansion/
│   ├── base.py              # QueryExpander Protocol
│   ├── hyde.py              # Hypothetical Document Embeddings
│   ├── synonym.py           # Domain synonym injection
│   └── decomposition.py     # Multi-sub-query decomposition
│
├── mocks/
│   ├── vertex_embedding.py  # Drop-in for TextEmbeddingModel (gecko)
│   └── vertex_generative.py # Drop-in for GenerativeModel (Gemini)
│
├── benchmark/
│   ├── corpus.py            # 10 GCP paragraphs + relevance labels (frozen)
│   ├── queries.py           # 5 evaluation queries + ground truth (frozen)
│   ├── metrics.py           # MRR@3, NDCG@3, P@3, latency, cosine_drift
│   ├── runner.py            # End-to-end orchestrator
│   └── report.py            # Renders JSON + Markdown reports
│
├── tests/
│   ├── conftest.py          # FakeEmbeddingModel + shared fixtures
│   ├── test_ingestion.py    # RAGEngine.ingest() unit tests
│   ├── test_retrieval.py    # Strategy A, B, Hybrid end-to-end
│   ├── test_mocks.py        # SDK signature parity (no GCP account needed)
│   ├── test_metrics.py      # MRR/NDCG/P@3/drift with known expected values
│   └── test_benchmark.py    # Parametrised: MRR > 0.5 for non-adversarial queries
│
├── benchmark_results.json   # Generated output — do not delete
├── retrieval_benchmark.md   # Generated output — do not delete
└── MIGRATION_GUIDE.md       # Vertex AI Matching Engine migration path
```

---

## Configuration

All parameters are in `config.yaml` — no hardcoded paths anywhere:

```yaml
embedding:
  model_name: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  dimension: 768

vector_store:
  type: faiss          # or "numpy" for environments without faiss-cpu

retrieval:
  top_k: 5
  latency_runs: 5      # median of N warm runs via time.perf_counter()

strategy_b:
  default_mode: good_expansion

hybrid:
  rrf_k: 60

output:
  results_json: benchmark_results.json
  report_md: retrieval_benchmark.md
```

---

## Architecture

```
config.yaml
    └─► RAGEngine.from_config()
            ├─► LocalEmbeddingModel   (768-dim gecko simulation)
            └─► RAGEngine.ingest(corpus)
                    ├─► FAISSVectorStore  (IndexFlatIP + L2 norm = cosine)
                    └─► BM25Store
                            │
                            ├─► Strategy A:  embed_query → FAISS.search
                            │
                            ├─► Strategy B:  MockGenerativeModel.expand()
                            │       ├─► good_expansion   → improved recall
                            │       ├─► noisy_expansion  → degraded recall
                            │       └─► null_expansion   → identical to A
                            │       embed_query(expanded) → FAISS.search
                            │       cosine_drift(orig_emb, expanded_emb)
                            │
                            └─► Hybrid: FAISS + BM25 → RRF(k=60)
                                    │
                                    └─► MRR@3 / NDCG@3 / P@3 / latency_ms
                                            └─► benchmark_results.json
                                                retrieval_benchmark.md
```

---

## Key Results (from latest run)

| Query | Strategy A NDCG | B_good NDCG | B_noisy NDCG | Insight |
|-------|----------------|-------------|--------------|---------|
| q1: peak load | 0.648 | **1.000** +54% | 0.500 | B_good improves ranking |
| q2: BigQuery cost | 1.000 | 1.000 | **0.307** -69% | Noisy expansion kills precision |
| q3: Vertex AI latency | 1.000 | 1.000 | **0.387** -61% | Domain noise shifts embedding far |
| q4: CAP theorem (adv) | 0.000 | 0.000 | 0.000 | Honest: corpus doesn't cover CAP |
| q5: PubSub flow ctrl | 1.000 | 1.000 | 1.000 | River/flow shares embedding space |

---

## Similarity Metric Selection: Cosine vs Euclidean

### Why Cosine?

Sentence embeddings encode meaning in **direction**, not magnitude. Two paraphrases
of the same sentence can have different norms depending on length and punctuation.
Cosine similarity ignores magnitude:

```
cosine(a, b) = dot(a, b) / (||a|| · ||b||)
```

Euclidean distance `||a − b||` conflates direction with scale, causing semantically
identical but differently-normed embeddings to appear distant.

### Why L2-normalisation + FAISS `IndexFlatIP` (not scipy)?

For L2-unit vectors, inner product equals cosine similarity:

```
||a|| = ||b|| = 1   =>   dot(a, b) = cosine(a, b)
```

Pre-normalising the corpus once at ingest time means every query is a single
fast matrix multiply — no per-query division. This is faster than `scipy.cdist`
and avoids Python-level loops:

```python
# What this project does
faiss.normalize_L2(corpus_embeddings)       # once at ingest
index = faiss.IndexFlatIP(dim)
index.search(query_normalised, top_k)       # fast BLAS matmul

# What we avoid
from scipy.spatial.distance import cdist
cdist(query, corpus, metric='cosine')       # Python loop, no GPU path
```

### Model Selection: Local Gecko Simulation

`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` was selected as the
local simulation of Vertex AI textembedding-gecko. Both are 768-dimensional and
multilingual. In production this mock is replaced by a one-line swap to
`vertexai.language_models.TextEmbeddingModel` — see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

**Note:** The assessment specified sentence-transformers as local simulation of
textembedding-gecko. `paraphrase-multilingual-mpnet-base-v2` was selected as it
shares gecko's 768-dim multilingual architecture. In production the one-line swap is:
`TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`.

The `EmbeddingModel` Protocol exposes `embed_documents()` and `embed_query()` as
separate methods to support future asymmetric models (such as BGE or E5). For the
current gecko-equivalent model, both methods encode identically — gecko does not
require a query prefix.

---

## Vertex AI Migration

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for:

- One-line swap: `MockTextEmbeddingModel` → real `TextEmbeddingModel`
- FAISS `IndexFlatIP` → Vertex AI Matching Engine (`COSINE_DISTANCE`)
- Latency delta: local ~30 ms vs Matching Engine ~10–50 ms + LLM expansion ~500–2000 ms
- Batch ingestion with `upsert_datapoints()`
- Cost model overview

---

## Metrics Reference

| Metric | Definition | Cutoff |
|--------|-----------|--------|
| MRR | 1 / rank of first relevant result | @3 |
| NDCG | DCG / IDCG; DCG = Σ rel_i/log₂(rank_i+1) | @3 |
| P@3 | (# relevant in top 3) / 3 | @3 |
| Latency | Median of 5 warm runs via `time.perf_counter()` | — |
| Cosine Drift | 1 − cosine_similarity(embed(q_orig), embed(q_exp)) | — |

Relevance labels: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant.
All labels are hand-assigned in `benchmark/queries.py` — never inferred from model output.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` or set `vector_store.type: numpy` in `config.yaml` |
| Slow first run | Normal — embedding model downloading (~1.4 GB). Subsequent runs are fast. |
| `UnicodeEncodeError` on Windows | Run from Windows Terminal or set `PYTHONIOENCODING=utf-8` |
| Port/permission error with Docker | Run `docker compose down` then retry |
| pytest fails with import error | Ensure you are in the project root and venv is activated |
