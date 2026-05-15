"""HyDE (Hypothetical Document Embeddings) query expander.

Instead of embedding the question, HyDE generates a hypothetical answer and
embeds that.  The idea: a document-shaped string is closer in embedding space
to the real documents than a question-shaped string is.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels", ACL 2023.
"""
from __future__ import annotations

from expansion.base import QueryExpander  # noqa: F401  (used for type-checking)


_HYPOTHETICAL_ANSWERS: dict[str, str] = {
    "How does the system handle peak load?": (
        "The system handles peak load by automatically scaling container instances "
        "using Cloud Run autoscaling and GKE cluster autoscaler. When traffic "
        "increases, additional instances are provisioned within seconds. The "
        "minimum-instances setting keeps warm replicas to avoid cold-start latency "
        "during burst traffic events."
    ),
    "What are the cost implications of BigQuery slot reservations?": (
        "BigQuery slot reservations allow organisations to commit to a fixed number "
        "of processing slots at a lower cost than on-demand pricing. One-year and "
        "three-year slot commitments offer further discounts. Unused slots can be "
        "shared across projects, reducing per-query cost for teams with variable "
        "analytical workloads."
    ),
    "How does Vertex AI manage model serving latency?": (
        "Vertex AI Online Prediction endpoints manage serving latency by deploying "
        "models on dedicated GPU/TPU accelerators and autoscaling replicas based on "
        "request volume. Traffic splitting spreads load across model versions. "
        "Scaling from zero means idle endpoints consume no compute until a prediction "
        "request arrives."
    ),
    "Can a distributed system guarantee consistency, availability, and partition tolerance simultaneously?": (
        "According to the CAP theorem, a distributed system can guarantee at most "
        "two of the three properties: consistency, availability, and partition "
        "tolerance. In practice, systems choose a trade-off, such as CP "
        "(HBase) or AP (Cassandra), depending on their failure model."
    ),
    "PubSub flow control": (
        "Cloud Pub/Sub flow control uses MaxOutstandingMessages and MaxOutstandingBytes "
        "settings to limit how many unacknowledged messages a subscriber holds at once. "
        "When the limit is reached, the client pauses delivery, applying backpressure "
        "upstream through lease management."
    ),
}


class HyDEExpander:
    """Expands a query by generating a hypothetical answer document.

    Satisfies the QueryExpander Protocol without inheriting from it.
    In production, the hypothetical answer would come from a live LLM call;
    here it is pre-computed for the benchmark corpus.
    """

    def expand(self, query: str) -> str:
        """Return a hypothetical answer to embed instead of the question.

        Args:
            query: The original user question.

        Returns:
            A document-shaped string that would answer the query.  Falls back
            to a generic template if the query is not in the lookup table.
        """
        return _HYPOTHETICAL_ANSWERS.get(
            query,
            f"A detailed technical answer explaining: {query}",
        )
