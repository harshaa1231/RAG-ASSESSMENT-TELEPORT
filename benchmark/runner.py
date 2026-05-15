"""End-to-end benchmark runner.

Usage:
    python -m benchmark.runner

Orchestrates all strategies across all 5 queries, captures real timings,
runs Strategy B in all 3 mock modes, and exports both output files.

After engine.benchmark() returns QueryMetrics objects, this module annotates
each Strategy B result with the actual expanded query text so evaluators can
verify what the mock generative model rewrote the query to before retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path

from benchmark.corpus import CORPUS
from benchmark.queries import QUERIES
from core.rag_engine import RAGEngine
from mocks.vertex_generative import MockGenerativeModel

# Must match the prompt template in retrieval/strategy_b.py exactly
_EXPANSION_PROMPT = (
    "Rewrite the following search query to improve document retrieval. "
    "Return only the rewritten query, no explanation.\n\nQuery: {query}"
)


def _annotate_expanded_texts(results: list) -> None:
    """Add expanded_query_text as a dynamic attribute to each QueryMetrics object.

    Strategy A and Hybrid results receive None.
    Strategy B results receive the actual rewritten query string that was used
    for retrieval — the key piece of information for evaluators to verify that
    expansion actually occurred.

    Args:
        results: List of QueryMetrics objects from engine.benchmark().
    """
    for qm in results:
        if qm.expansion_mode is not None:
            expander = MockGenerativeModel("gemini-1.5-flash", mode=qm.expansion_mode)
            prompt = _EXPANSION_PROMPT.format(query=qm.query_text)
            qm.expanded_query_text = expander.expand(prompt)
        else:
            qm.expanded_query_text = None


def run(config_path: str | Path = "config.yaml") -> None:
    """Execute the full benchmark pipeline.

    Args:
        config_path: Path to config.yaml relative to the working directory.
    """
    print("=" * 60)
    print("RAG Benchmark — Strategy A vs Strategy B vs Hybrid")
    print("=" * 60)

    print("\n[1/4] Loading config and embedding model...")
    engine = RAGEngine.from_config(config_path)

    print("[2/4] Ingesting corpus...")
    engine.ingest(CORPUS)
    print(f"      Indexed {len(CORPUS)} chunks into dense + sparse stores")

    print("[3/4] Running benchmark (5 queries × 5 strategies)...")
    results = engine.benchmark(QUERIES, strategies=["A", "B", "hybrid"])

    # Annotate Strategy B results with expanded query text for the report
    _annotate_expanded_texts(results)

    by_query: dict[str, list] = {}
    for r in results:
        by_query.setdefault(r.query_id, []).append(r)

    for qid, qresults in sorted(by_query.items()):
        query_text = qresults[0].query_text
        ellipsis = "..." if len(query_text) > 55 else ""
        print(f'\n  {qid}: "{query_text[:55]}{ellipsis}"')
        for qm in qresults:
            drift_str = f"  drift={qm.cosine_drift:.3f}" if qm.cosine_drift is not None else ""
            expanded_str = ""
            exp_text = getattr(qm, "expanded_query_text", None)
            if exp_text:
                preview = exp_text[:40]
                expanded_str = f'  expanded="{preview}..."'
            print(
                f"    {qm.strategy:<22}  "
                f"MRR={qm.mrr:.3f}  NDCG={qm.ndcg:.3f}  "
                f"P@3={qm.p3:.3f}  lat={qm.latency_ms:.1f}ms"
                f"{drift_str}{expanded_str}"
            )

    print("\n[4/4] Exporting reports...")
    # Call ReportWriter directly so expanded_query_text annotations are included.
    # engine.export_report() does not carry these dynamic attributes.
    from benchmark.report import ReportWriter
    from benchmark.dashboard import write_dashboard
    config = engine.config
    writer = ReportWriter(results, CORPUS)
    writer.write_json(Path(config.output.results_json))
    writer.write_markdown(Path(config.output.report_md))
    write_dashboard(Path("dashboard.html"), results)
    print("      [OK] benchmark_results.json")
    print("      [OK] retrieval_benchmark.md")
    print("      [OK] dashboard.html  ← open in any browser, no server needed")
    print("\nDone.")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config)
