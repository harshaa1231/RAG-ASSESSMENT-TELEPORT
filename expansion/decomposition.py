"""Multi-sub-query decomposition expander.

Breaks a complex query into simpler sub-queries.  The caller (retrieval layer)
can retrieve separately for each sub-query and union the results; this expander
returns a single concatenated string suitable for unified embedding or for
splitting on the newline separator.
"""
from __future__ import annotations

_DECOMPOSITIONS: dict[str, list[str]] = {
    "How does the system handle peak load?": [
        "How does Cloud Run autoscaling work?",
        "How does GKE cluster autoscaler provision nodes?",
        "How are traffic spikes handled in Google Cloud?",
    ],
    "What are the cost implications of BigQuery slot reservations?": [
        "What are BigQuery slot reservations?",
        "How does BigQuery on-demand pricing compare to slot pricing?",
        "What discount do long-term BigQuery slot commitments offer?",
    ],
    "How does Vertex AI manage model serving latency?": [
        "What accelerators does Vertex AI use for inference?",
        "How does Vertex AI autoscale prediction endpoints?",
        "What is traffic splitting in Vertex AI?",
    ],
    "Can a distributed system guarantee consistency, availability, and partition tolerance simultaneously?": [
        "What is the CAP theorem?",
        "What are consistency and availability trade-offs in distributed systems?",
        "What does partition tolerance mean in distributed databases?",
    ],
    "PubSub flow control": [
        "What is MaxOutstandingMessages in Pub/Sub?",
        "How does Pub/Sub apply backpressure to subscribers?",
        "What is lease management in Cloud Pub/Sub?",
    ],
}

SUB_QUERY_SEPARATOR = "\n"


class DecompositionExpander:
    """Decomposes a complex query into multiple focused sub-queries.

    Returns a single string with sub-queries joined by newlines.
    The retrieval layer can split on SUB_QUERY_SEPARATOR to retrieve per
    sub-query and union the results.

    Satisfies the QueryExpander Protocol without inheriting from it.
    """

    def expand(self, query: str) -> str:
        """Return newline-separated sub-queries for the input.

        Args:
            query: The original composite query.

        Returns:
            Sub-queries joined by newlines, or the original query if no
            decomposition is defined.
        """
        subs = _DECOMPOSITIONS.get(query)
        if subs:
            return SUB_QUERY_SEPARATOR.join(subs)
        return query

    def expand_to_list(self, query: str) -> list[str]:
        """Return sub-queries as a list for multi-retrieval patterns.

        Args:
            query: The original composite query.

        Returns:
            List of sub-query strings.
        """
        return _DECOMPOSITIONS.get(query, [query])
