"""Mock that mirrors vertexai.generative_models.GenerativeModel exactly.

The real SDK interface:
    from vertexai.generative_models import GenerativeModel
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("prompt text")  # -> GenerationResponse
    response.text                                       # -> str

    # Older text-model API:
    from vertexai.language_models import TextGenerationModel
    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict("prompt text")            # -> TextGenerationResponse
    response.text                                       # -> str

Both APIs are mirrored here.  Mode is set at construction time.

Mock modes
──────────
good_expansion  Expands the query with semantically aligned GCP-domain terms.
                Expected effect: equal or better retrieval quality vs Strategy A.
noisy_expansion Returns a query containing only off-domain noise terms with NO
                semantic overlap with the relevant chunks.
                Expected effect: measurable retrieval degradation vs Strategy A.
null_expansion  Returns the original query unchanged — always, for every query.
                Expected effect: identical results to Strategy A.

Expansion dict
──────────────
Keys are the exact query texts used in benchmark/queries.py.
Good expansions reinforce the GCP terminology already present in the query.
Noisy expansions are deliberately chosen from completely unrelated domains
(medieval history, cooking, sports, agriculture) to guarantee semantic
distance from every corpus chunk and produce measurably lower retrieval metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

EXPANSION_MODES = ("good_expansion", "noisy_expansion", "null_expansion")

# ---------------------------------------------------------------------------
# Expansions keyed by EXACT query text from benchmark/queries.py.
#
# Noisy expansion design rule: must contain ZERO words that appear in the
# FakeEmbeddingModel keyword groups (cloud run, autoscal, traffic, peak,
# scale, burst, bigquery, slot reservation, pub/sub, pubsub, flow control,
# backpressure, outstanding, vertex ai, serving, endpoint, inference, latency,
# gke, kubernetes, cluster autoscal, node pool).  This guarantees zero-vector
# embeddings in tests → clear degradation signal → test assertions pass.
# ---------------------------------------------------------------------------

_EXPANSIONS: dict[str, dict[str, str]] = {
    # ── q1 ─────────────────────────────────────────────────────────────────
    "How does the system handle peak load?": {
        "good": (
            "Cloud Run concurrency limits request queuing autoscaling thresholds "
            "horizontal scaling instance warm-up latency under high traffic volume "
            "load balancer distribution GKE cluster autoscaler node provisioning"
        ),
        "noisy": (
            "blockchain proof-of-work medieval history painting art sculpture "
            "cooking recipe dinner invitation completely unrelated content"
        ),
        "null": "How does the system handle peak load?",
    },

    # ── q2 ─────────────────────────────────────────────────────────────────
    "What are the cost implications of BigQuery slot reservations?": {
        "good": (
            "BigQuery slot reservation flex slots capacity commitment annual "
            "three-year discount pricing on-demand versus committed use billing "
            "per-slot workload cost analysis reservation assignment organisation"
        ),
        "noisy": (
            "economy inflation mortgage real-estate cooking recipe medieval "
            "history painting art sculpture music concert completely off topic"
        ),
        "null": "What are the cost implications of BigQuery slot reservations?",
    },

    # ── q3 ─────────────────────────────────────────────────────────────────
    "How does Vertex AI manage model serving latency?": {
        "good": (
            "Vertex AI Online Prediction endpoint autoscaling GPU TPU accelerator "
            "replica inference latency SLO traffic splitting deployment model "
            "serving response time throughput warm-up cold start"
        ),
        "noisy": (
            "sports tournament volleyball beach football match weather forecast "
            "agriculture irrigation gardening botany philosophy astronomy geology"
        ),
        "null": "How does Vertex AI manage model serving latency?",
    },

    # ── q4 (adversarial) ───────────────────────────────────────────────────
    "Can a distributed system guarantee consistency, availability, and partition tolerance simultaneously?": {
        "good": (
            "CAP theorem distributed database trade-off consistency availability "
            "partition tolerance eventual strong linearizable Spanner Bigtable "
            "network partition distributed systems choice two-of-three"
        ),
        "noisy": (
            "hairstyle salon beauty appointment party planning event management "
            "cooking pastry baking medieval history poetry literature completely "
            "off domain and unrelated to distributed systems"
        ),
        "null": "Can a distributed system guarantee consistency, availability, and partition tolerance simultaneously?",
    },

    # ── q5 ─────────────────────────────────────────────────────────────────
    "PubSub flow control": {
        "good": (
            "Cloud Pub/Sub flow control MaxOutstandingMessages MaxOutstandingBytes "
            "subscriber backpressure lease management acknowledgement deadline "
            "message delivery throughput rate limiting push pull subscription"
        ),
        "noisy": (
            "river dam irrigation agriculture gardening botany medieval history "
            "painting cooking recipe philosophy astronomy geology poetry literature"
        ),
        "null": "PubSub flow control",
    },
}


# ---------------------------------------------------------------------------
# Mirror data classes
# ---------------------------------------------------------------------------


@dataclass
class MockGenerationResponse:
    """Mirrors vertexai.generative_models.GenerationResponse."""

    text: str


@dataclass
class MockPredictResponse:
    """Mirrors vertexai.language_models.TextGenerationResponse from predict()."""

    text: str


# ---------------------------------------------------------------------------
# GenerativeModel mirror (Gemini API)
# ---------------------------------------------------------------------------


class MockGenerativeModel:
    """Drop-in replacement for vertexai.generative_models.GenerativeModel.

    Supports good_expansion, noisy_expansion, and null_expansion modes.
    Mode is injected at construction — in production this becomes a live LLM call.
    """

    def __init__(self, model_name: str, mode: str = "good_expansion") -> None:
        """Initialise the mock.

        Args:
            model_name: Vertex AI model identifier (stored for parity, not used).
            mode: One of "good_expansion", "noisy_expansion", "null_expansion".
        """
        if mode not in EXPANSION_MODES:
            raise ValueError(f"mode must be one of {EXPANSION_MODES}, got {mode!r}")
        self._model_name = model_name
        self.mode = mode

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        mode: str = "good_expansion",
    ) -> "MockGenerativeModel":
        """Factory mirroring the real SDK pattern."""
        return cls(model_name=model_name, mode=mode)

    def generate_content(self, prompt: str) -> MockGenerationResponse:
        """Generate expanded query text.

        The prompt must end with "Query: <original query text>" so the mock can
        extract the original query and look up its pre-defined expansion.

        Args:
            prompt: Full prompt string sent to the generative model.

        Returns:
            MockGenerationResponse with .text containing the expanded query.
        """
        original_query = self._extract_query(prompt)
        expanded = self._expand(original_query)
        return MockGenerationResponse(text=expanded)

    def predict(self, prompt: str, **kwargs: object) -> MockPredictResponse:
        """Older TextGenerationModel.predict() API — mirrors real signature exactly.

        Args:
            prompt: Text prompt.
            **kwargs: Accepted for signature parity (temperature, max_output_tokens, etc.)

        Returns:
            MockPredictResponse with .text attribute.
        """
        original_query = self._extract_query(prompt)
        expanded = self._expand(original_query)
        return MockPredictResponse(text=expanded)

    def expand(self, prompt: str) -> str:
        """Satisfy the QueryExpander Protocol used by ExpandedQueryRetriever.

        Delegates to generate_content() and returns response.text so this mock
        can be passed anywhere a QueryExpander is expected.
        """
        return self.generate_content(prompt).text

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_query(prompt: str) -> str:
        """Extract original query from prompt formatted as '...Query: <text>'."""
        if "Query:" in prompt:
            return prompt.split("Query:")[-1].strip()
        if "Question:" in prompt:
            return prompt.split("Question:")[-1].strip()
        return prompt.strip()

    def _expand(self, query: str) -> str:
        """Look up expansion from dict; apply mode-appropriate fallback if missing."""
        # null_expansion always returns the original query — regardless of whether
        # the query is in the dict.
        if self.mode == "null_expansion":
            return _EXPANSIONS.get(query, {}).get("null", query)

        entry = _EXPANSIONS.get(query)
        if entry is None:
            # Fallback for unseen queries: append generic scalability terms.
            # This preserves original keywords so FakeEmbeddingModel still works.
            return query + " system performance scalability reliability"

        mode_key = "good" if self.mode == "good_expansion" else "noisy"
        return entry[mode_key]
