"""Retrieval quality metrics: MRR@k, NDCG@k, Precision@k, latency, cosine_drift.

All metrics are computed from real relevance labels — nothing is mocked or fabricated.
Latency is always the median of 5 warm runs via time.perf_counter().
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single result from a retrieval call."""

    chunk_id: str
    score: float
    relevance: int  # ground-truth label: 0, 1, or 2
    rank: int       # 1-indexed position in the result list


@dataclass
class QueryMetrics:
    """Aggregated metrics for one (query, strategy) pair."""

    query_id: str
    query_text: str
    strategy: str
    expansion_mode: str | None
    retrieved: list[RetrievedChunk]
    mrr: float
    ndcg: float
    p3: float
    latency_ms: float
    cosine_drift: float | None


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def mrr_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int = 3) -> float:
    """Mean Reciprocal Rank at k.

    Finds the first relevant result (relevance > 0) within the top-k and returns
    1 / rank.  Returns 0 if no relevant result appears in the top-k window.

    Args:
        retrieved_ids: Chunk IDs in retrieval order (best first).
        relevance: Ground-truth dict mapping chunk_id to relevance score.
        k: Cutoff position.

    Returns:
        MRR value in [0, 1].
    """
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if relevance.get(chunk_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int = 3) -> float:
    """Normalised Discounted Cumulative Gain at k.

    DCG uses the formula: sum(rel_i / log2(rank_i + 1)) for ranks 1..k.
    IDCG is the DCG of the perfect ranking (ideal sorted by relevance descending).
    NDCG = DCG / IDCG.  Returns 0 when all ground-truth labels are 0 (adversarial).

    Args:
        retrieved_ids: Chunk IDs in retrieval order.
        relevance: Ground-truth relevance labels.
        k: Cutoff position.

    Returns:
        NDCG value in [0, 1].
    """
    dcg = sum(
        relevance.get(cid, 0) / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int = 3) -> float:
    """Precision at k: fraction of top-k results that are relevant (relevance > 0).

    Args:
        retrieved_ids: Chunk IDs in retrieval order.
        relevance: Ground-truth relevance labels.
        k: Cutoff position.

    Returns:
        P@k value in [0, 1].
    """
    relevant = sum(
        1 for cid in retrieved_ids[:k] if relevance.get(cid, 0) > 0
    )
    return relevant / k


def cosine_drift(
    original_embedding: np.ndarray,
    expanded_embedding: np.ndarray,
) -> float:
    """Cosine distance between original and expanded query embeddings.

    cosine_drift = 1 - cosine_similarity(v_original, v_expanded).
    A drift of 0 means the expansion didn't change query direction; 1 means
    the expanded query is orthogonal to the original.

    Args:
        original_embedding: Raw (un-normalised) embedding of original query.
        expanded_embedding: Raw embedding of expanded query.

    Returns:
        Cosine distance in [0, 2] (typically [0, 1] for non-negative embeddings).
    """
    v1 = original_embedding.flatten().astype(np.float64)
    v2 = expanded_embedding.flatten().astype(np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1.0  # Maximum drift — one vector is degenerate
    sim = np.dot(v1 / n1, v2 / n2)
    return float(1.0 - np.clip(sim, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_latency_ms(fn: Callable[[], object], runs: int = 5) -> float:
    """Run fn `runs` times and return the median wall-clock time in milliseconds.

    The first call serves as a warm-up that is included in the sample.
    Uses time.perf_counter() for high-resolution timing.

    Args:
        fn: Zero-argument callable to time.
        runs: Number of timed repetitions (default 5, never fewer).

    Returns:
        Median latency in milliseconds.
    """
    timings: list[float] = []
    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return float(np.median(timings))


# ---------------------------------------------------------------------------
# Composite metric builder
# ---------------------------------------------------------------------------


def build_query_metrics(
    query_id: str,
    query_text: str,
    strategy: str,
    expansion_mode: str | None,
    retrieved_pairs: list[tuple[str, float]],
    relevance: dict[str, int],
    latency_ms: float,
    original_embedding: np.ndarray | None = None,
    expanded_embedding: np.ndarray | None = None,
) -> QueryMetrics:
    """Assemble a QueryMetrics object from raw retrieval output.

    Args:
        query_id: Query identifier string.
        query_text: Original query text.
        strategy: e.g. "A", "B_good", "hybrid".
        expansion_mode: None for Strategy A; "good_expansion" etc. for B.
        retrieved_pairs: (chunk_id, score) list in retrieval order.
        relevance: Ground-truth chunk -> relevance mapping.
        latency_ms: Pre-measured median latency.
        original_embedding: Used for cosine_drift if provided.
        expanded_embedding: Used for cosine_drift if provided.

    Returns:
        Fully populated QueryMetrics.
    """
    retrieved_ids = [cid for cid, _ in retrieved_pairs]
    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            score=score,
            relevance=relevance.get(cid, 0),
            rank=rank,
        )
        for rank, (cid, score) in enumerate(retrieved_pairs, start=1)
    ]
    drift: float | None = None
    if original_embedding is not None and expanded_embedding is not None:
        drift = cosine_drift(original_embedding, expanded_embedding)

    return QueryMetrics(
        query_id=query_id,
        query_text=query_text,
        strategy=strategy,
        expansion_mode=expansion_mode,
        retrieved=chunks,
        mrr=mrr_at_k(retrieved_ids, relevance),
        ndcg=ndcg_at_k(retrieved_ids, relevance),
        p3=precision_at_k(retrieved_ids, relevance),
        latency_ms=latency_ms,
        cosine_drift=drift,
    )
