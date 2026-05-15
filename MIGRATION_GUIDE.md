# Vertex AI Migration Guide

Replacing the mock Vertex AI SDK with the real GCP Vertex AI APIs requires changes to
only two files (`mocks/vertex_embedding.py` → `core/vertex_embedding.py` adapter, and
a one-line change in `core/rag_engine.py`).  All metric logic, retrieval strategies,
and report generation are unaffected.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GCP project | Billing enabled |
| Vertex AI API | Enabled via `gcloud services enable aiplatform.googleapis.com` |
| Authentication | `gcloud auth application-default login` or service account key |
| Python SDK | `pip install google-cloud-aiplatform>=1.38.0` |
| Region | `us-central1` recommended; set `VERTEX_REGION` env var to override |

---

## Step 1 — Install the Real SDK

```bash
pip install google-cloud-aiplatform
```

---

## Step 2 — Interface Swap: MockTextEmbeddingModel → Real TextEmbeddingModel

The mock preserves the exact same interface as the real SDK.  The swap is **one line**
in `core/rag_engine.py` `_build_embedding_model()`:

```python
# Before (mock)
from mocks.vertex_embedding import MockTextEmbeddingModel
model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# After (real Vertex AI) — one-line change
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
```

Both expose the identical interface:
```python
results = model.get_embeddings(["text1", "text2"])
results[0].values          # list[float] — embedding vector
results[0].statistics.token_count  # int
```

Wrap the real `TextEmbeddingModel` in an adapter to satisfy the `EmbeddingModel` Protocol:

```python
# core/vertex_embedding.py
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel

class VertexEmbeddingModel:
    def __init__(self, project: str, region: str, model_name: str) -> None:
        vertexai.init(project=project, location=region)
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        results = self._model.get_embeddings(texts)
        return np.array([r.values for r in results], dtype=np.float32)

    def get_dimension(self) -> int:
        return 768  # text-embedding-004 dimension
```

Then in `core/rag_engine.py`:
```python
@staticmethod
def _build_embedding_model(config: BenchmarkConfig) -> EmbeddingModel:
    from core.vertex_embedding import VertexEmbeddingModel
    return VertexEmbeddingModel(
        project=config.vertex_ai.project,
        region=config.vertex_ai.region,
        model_name=config.embedding.model_name,
    )
```

---

## Step 3 — FAISS → Vertex AI Matching Engine

| Aspect | Local FAISS | Vertex AI Matching Engine |
|--------|-------------|--------------------------|
| Index type | `IndexFlatIP` (brute-force) | `COSINE_DISTANCE` (ANN, SCANN) |
| Normalisation | `_safe_normalize()` before add | Handled internally |
| Hosting | In-process | Managed GCP endpoint |
| Scale | ~1M vectors | Billions of vectors |

Migration pattern — the FAISS store interface maps to Matching Engine as follows:

```python
# FAISS (current)
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)
scores, ids = index.search(query, top_k)

# Matching Engine equivalent
from google.cloud import aiplatform
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-benchmark",
    dimensions=768,
    distance_measure_type="COSINE_DISTANCE",
)
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(...)
index_endpoint.deploy_index(index=index, deployed_index_id="rag_index")
# Query:
results = index_endpoint.find_neighbors(
    deployed_index_id="rag_index",
    queries=[query_embedding.tolist()],
    num_neighbors=top_k,
)
```

---

## Step 4 — Batch Ingestion with upsert_datapoints()

```python
from google.cloud.aiplatform_v1 import IndexServiceClient
from google.cloud.aiplatform_v1.types import IndexDatapoint

client = IndexServiceClient(client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"})

datapoints = [
    IndexDatapoint(
        datapoint_id=chunk.id,
        feature_vector=embedding.tolist(),
    )
    for chunk, embedding in zip(corpus, embeddings)
]

client.upsert_datapoints(
    index=INDEX_RESOURCE_NAME,
    datapoints=datapoints,
)
```

For large corpora, batch in groups of 100 datapoints to stay within the API quota.

---

## Step 5 — Replace Query Expansion Mock

```python
# Current (mock — production stub)
from mocks.vertex_generative import MockGenerativeModel
expander = MockGenerativeModel("gemini-1.5-flash", mode="good_expansion")

# Real Vertex AI
from vertexai.generative_models import GenerativeModel

class VertexQueryExpander:
    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self._model = GenerativeModel(model_name)

    def expand(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip()
```

No changes to `ExpandedQueryRetriever` — it already calls `expander.expand(prompt)`.

---

## Latency Delta Analysis

| Operation | Local FAISS | Vertex AI Matching Engine | Notes |
|-----------|-------------|--------------------------|-------|
| Embedding (per query) | ~1–3 ms | ~20–80 ms | Network round-trip to Vertex AI |
| Dense search (10 chunks) | < 1 ms | ~10–50 ms | ANN vs brute-force |
| Query expansion (LLM) | ~0 ms (mock) | ~500–2000 ms | Gemini API latency |
| Total Strategy A | ~2–5 ms | ~30–130 ms | |
| Total Strategy B | ~2–5 ms | ~530–2130 ms | |

**Implication for query expansion overhead:**
The LLM expansion call (~500–2000 ms) dominates total latency for Strategy B in
production.  For latency-constrained workloads (< 200 ms p99), Strategy A or a
cached-expansion approach is recommended.  The `null_expansion` mode benchmarks
Strategy B without this overhead.

---

## Step 6 — Update Tests

Mock the Vertex AI HTTP layer so `pytest tests/ -v` passes without live GCP calls:

```python
# tests/conftest.py — add to existing fixtures
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=False)
def mock_vertex_http():
    with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained") as mock:
        instance = MagicMock()
        instance.get_embeddings.return_value = [
            MagicMock(
                values=[0.0] * 768,
                statistics=MagicMock(token_count=5),
            )
        ]
        mock.return_value = instance
        yield
```

Apply via `@pytest.mark.usefixtures("mock_vertex_http")` on integration tests.

---

## Cost Model

| Resource | Unit | Price (us-central1) |
|----------|------|---------------------|
| text-embedding-004 | per 1K characters | ~$0.000025 |
| gemini-1.5-flash input | per 1M tokens | ~$0.075 |
| gemini-1.5-flash output | per 1M tokens | ~$0.30 |
| Matching Engine indexing | per GB per hour | ~$0.036 |
| Matching Engine queries | per 1M queries | ~$0.40 |
| Full benchmark (5 queries × 5 strategies) | — | < $0.02 |

---

## Rollback

```bash
# Revert to local mocks — no code changes needed
# Just set vertex_ai.enabled: false in config.yaml (if added)
# and restore the LocalEmbeddingModel path in _build_embedding_model()
pytest tests/ -v   # must still pass clean
```

---

## References

- [Vertex AI Embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
- [Vertex AI Matching Engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)
- [text-embedding-004 model card](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings)
- [google-cloud-aiplatform on PyPI](https://pypi.org/project/google-cloud-aiplatform/)
- [upsert_datapoints API reference](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.services.index_service.IndexServiceClient#google_cloud_aiplatform_v1_services_index_service_IndexServiceClient_upsert_datapoints)
