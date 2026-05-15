"""Five evaluation queries with hand-assigned ground-truth relevance labels.

Relevance scale per chunk: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant.
Query set must not be modified after initial creation — it is the ground truth.

Query inventory
───────────────
q1  Required — peak load handling (multi-topic: Cloud Run + GKE)
q2  Required — BigQuery cost/slot reservations
q3  Required — Vertex AI serving latency
q4  Adversarial — CAP theorem (not covered by the GCP corpus)
q5  Keyword-exact — "PubSub flow control"
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Query:
    """A single evaluation query with ground-truth relevance per chunk."""

    id: str
    text: str
    relevance: dict[str, int]  # chunk_id -> {0, 1, 2}
    is_adversarial: bool = False


QUERIES: list[Query] = [
    # ── q1: peak load handling ─────────────────────────────────────────────
    Query(
        id="q1",
        text="How does the system handle peak load?",
        relevance={
            "chunk_0": 2,  # Cloud Run autoscaling — directly answers peak load
            "chunk_1": 2,  # Cloud Run min-instances — cold-start + peak
            "chunk_2": 0,  # BigQuery slots — unrelated to peak load handling
            "chunk_3": 0,  # BigQuery slot reservations — unrelated
            "chunk_4": 1,  # PubSub flow control — indirectly limits overload
            "chunk_5": 1,  # PubSub backpressure — partially relevant
            "chunk_6": 1,  # Vertex AI endpoint autoscaling — partially relevant
            "chunk_7": 1,  # Vertex AI burst scaling — partially relevant
            "chunk_8": 2,  # GKE cluster autoscaler — directly answers peak load
            "chunk_9": 2,  # GKE node auto-provisioning — directly answers peak load
        },
    ),
    # ── q2: BigQuery cost implications ────────────────────────────────────
    Query(
        id="q2",
        text="What are the cost implications of BigQuery slot reservations?",
        relevance={
            "chunk_0": 0,
            "chunk_1": 0,
            "chunk_2": 2,  # BigQuery slot capacity and pricing models
            "chunk_3": 2,  # Slot reservation commitments and cost discounts
            "chunk_4": 0,
            "chunk_5": 0,
            "chunk_6": 0,
            "chunk_7": 0,
            "chunk_8": 0,
            "chunk_9": 0,
        },
    ),
    # ── q3: Vertex AI serving latency ─────────────────────────────────────
    Query(
        id="q3",
        text="How does Vertex AI manage model serving latency?",
        relevance={
            "chunk_0": 0,
            "chunk_1": 0,
            "chunk_2": 0,
            "chunk_3": 0,
            "chunk_4": 0,
            "chunk_5": 0,
            "chunk_6": 2,  # Vertex AI endpoints — low-latency serving
            "chunk_7": 2,  # GPU/TPU accelerators and replica autoscaling for latency
            "chunk_8": 0,
            "chunk_9": 0,
        },
    ),
    # ── q4: Adversarial — CAP theorem ─────────────────────────────────────
    Query(
        id="q4",
        text=(
            "Can a distributed system guarantee consistency, availability, "
            "and partition tolerance simultaneously?"
        ),
        relevance={
            # CAP theorem is not covered by the GCP paragraph corpus.
            # All relevance labels are 0 — this query is designed to expose
            # hallucination risk and measure the honest failure mode.
            "chunk_0": 0,
            "chunk_1": 0,
            "chunk_2": 0,
            "chunk_3": 0,
            "chunk_4": 0,
            "chunk_5": 0,
            "chunk_6": 0,
            "chunk_7": 0,
            "chunk_8": 0,
            "chunk_9": 0,
        },
        is_adversarial=True,
    ),
    # ── q5: Keyword-exact ─────────────────────────────────────────────────
    Query(
        id="q5",
        text="PubSub flow control",
        relevance={
            "chunk_0": 0,
            "chunk_1": 0,
            "chunk_2": 0,
            "chunk_3": 0,
            "chunk_4": 2,  # MaxOutstandingMessages — exact topic match
            "chunk_5": 2,  # Lease management + backpressure — exact topic match
            "chunk_6": 0,
            "chunk_7": 0,
            "chunk_8": 0,
            "chunk_9": 0,
        },
    ),
]
