"""Domain synonym injection expander.

Appends GCP-domain synonyms for recognised terms in the query, broadening
coverage without completely rewriting the original intent.
"""
from __future__ import annotations

# Term -> space-separated synonym string appended to the query
_SYNONYM_MAP: dict[str, str] = {
    "peak load": "traffic spike burst traffic high concurrency request surge",
    "scale": "autoscale horizontal scaling elasticity auto-provisioning",
    "cost": "pricing billing charges expenditure budget",
    "latency": "response time delay throughput performance serving time",
    "flow control": "backpressure rate limiting throttling message pacing",
    "slot": "processing unit compute capacity reservation",
    "serving": "inference prediction deployment endpoint",
    "cluster": "node pool kubernetes workload orchestration",
}


class SynonymExpander:
    """Appends domain-specific synonyms to recognised query terms.

    Satisfies the QueryExpander Protocol without inheriting from it.
    """

    def expand(self, query: str) -> str:
        """Append synonyms for any matched domain terms.

        Args:
            query: Original query string.

        Returns:
            Extended query with relevant synonyms appended.
        """
        additions: list[str] = []
        lower = query.lower()
        for term, synonyms in _SYNONYM_MAP.items():
            if term in lower:
                additions.append(synonyms)
        if additions:
            return f"{query} {' '.join(additions)}"
        return query
