"""Protocol definition for query expanders — no inheritance required."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryExpander(Protocol):
    """Structural interface for anything that rewrites a query string."""

    def expand(self, query: str) -> str:
        """Rewrite or augment the input query.

        Args:
            query: The original user query.

        Returns:
            A rewritten or augmented query string. For null expansion this is
            identical to the input.
        """
        ...
