"""RAGEngine — central orchestration class for the benchmark system.

Entry point:
    engine = RAGEngine.from_config('config.yaml')
    engine.ingest(corpus)
    results = engine.benchmark(queries, strategies=['A', 'B', 'hybrid'])
    engine.export_report()
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.corpus import Chunk
from benchmark.metrics import QueryMetrics, build_query_metrics, measure_latency_ms
from benchmark.queries import Query
from core.config import BenchmarkConfig
from embeddings.base import EmbeddingModel
from embeddings.sparse import BM25Store
from mocks.vertex_generative import MockGenerativeModel
from retrieval.hybrid import HybridRetriever
from retrieval.strategy_a import RawVectorRetriever
from retrieval.strategy_b import ExpandedQueryRetriever
from vector_store.base import VectorStore

if TYPE_CHECKING:
    pass

_STRATEGY_B_MODES = ("good_expansion", "noisy_expansion", "null_expansion")


class RAGEngine:
    """Orchestrates corpus ingestion, multi-strategy retrieval, and reporting.

    Dependencies are injected at construction time, enabling Protocol-based
    swap of any component (embedding model, vector store, expander) without
    changing this class.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        config: BenchmarkConfig,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.config = config
        self._corpus: list[Chunk] = []
        self._results: list[QueryMetrics] = []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path = "config.yaml") -> "RAGEngine":
        """Construct a RAGEngine from a YAML config file.

        Args:
            config_path: Path to config.yaml.

        Returns:
            Fully wired RAGEngine ready for ingest().
        """
        config = BenchmarkConfig.from_yaml(config_path)
        embedding_model = cls._build_embedding_model(config)
        vector_store = cls._build_vector_store(config)
        bm25_store = BM25Store()
        return cls(embedding_model, vector_store, bm25_store, config)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, corpus: list[Chunk]) -> None:
        """Encode and index all corpus chunks.

        Populates both the dense (FAISS/NumPy) and sparse (BM25) indexes.

        Args:
            corpus: List of Chunk objects from benchmark.corpus.
        """
        self._corpus = list(corpus)
        texts = [chunk.text for chunk in corpus]
        chunk_ids = [chunk.id for chunk in corpus]

        embeddings = self.embedding_model.embed_documents(texts)
        self.vector_store.add(embeddings, chunk_ids, texts)
        self.bm25_store.add(texts, chunk_ids)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        queries: list[Query],
        strategies: list[str] | None = None,
    ) -> list[QueryMetrics]:
        """Run all requested strategies over all queries and collect metrics.

        When 'B' is in strategies, all three mock modes are run automatically:
        good_expansion, noisy_expansion, and null_expansion.

        Args:
            queries: Evaluation queries with ground-truth relevance labels.
            strategies: Subset of ['A', 'B', 'hybrid'].  Defaults to all three.

        Returns:
            Flat list of QueryMetrics, one per (query × strategy/mode) pair.
        """
        if strategies is None:
            strategies = ["A", "B", "hybrid"]

        top_k = self.config.retrieval.top_k
        runs = self.config.retrieval.latency_runs
        all_results: list[QueryMetrics] = []

        for query in queries:
            if "A" in strategies:
                all_results.extend(self._run_strategy_a(query, top_k, runs))
            if "B" in strategies:
                all_results.extend(self._run_strategy_b(query, top_k, runs))
            if "hybrid" in strategies:
                all_results.extend(self._run_hybrid(query, top_k, runs))

        self._results = all_results
        return all_results

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------

    def export_report(self, output_dir: str | Path | None = None) -> None:
        """Write benchmark_results.json and retrieval_benchmark.md.

        Args:
            output_dir: Directory for output files.  Defaults to current
                        working directory (as specified in config.output).
        """
        from benchmark.report import ReportWriter
        writer = ReportWriter(self._results, self._corpus)
        base = Path(output_dir) if output_dir else Path(".")
        writer.write_json(base / self.config.output.results_json)
        writer.write_markdown(base / self.config.output.report_md)

    # ------------------------------------------------------------------
    # Per-strategy runners
    # ------------------------------------------------------------------

    def _run_strategy_a(
        self,
        query: Query,
        top_k: int,
        runs: int,
    ) -> list[QueryMetrics]:
        retriever = RawVectorRetriever(self.embedding_model, self.vector_store)

        def _call() -> tuple:
            return retriever.retrieve(query.text, top_k)

        pairs, q_emb = _call()  # warm-up + capture embedding
        latency = measure_latency_ms(lambda: retriever.retrieve(query.text, top_k), runs)

        return [
            build_query_metrics(
                query_id=query.id,
                query_text=query.text,
                strategy="A",
                expansion_mode=None,
                retrieved_pairs=pairs,
                relevance=query.relevance,
                latency_ms=latency,
                original_embedding=None,
                expanded_embedding=None,
            )
        ]

    def _run_strategy_b(
        self,
        query: Query,
        top_k: int,
        runs: int,
    ) -> list[QueryMetrics]:
        results: list[QueryMetrics] = []
        for mode in _STRATEGY_B_MODES:
            expander = MockGenerativeModel(
                model_name="gemini-1.5-flash",
                mode=mode,
            )
            retriever = ExpandedQueryRetriever(
                self.embedding_model,
                self.vector_store,
                expander,
                mode,
            )

            pairs, orig_emb, exp_emb, _ = retriever.retrieve(query.text, top_k)
            latency = measure_latency_ms(
                lambda: retriever.retrieve(query.text, top_k), runs
            )

            results.append(
                build_query_metrics(
                    query_id=query.id,
                    query_text=query.text,
                    strategy=f"B_{mode}",
                    expansion_mode=mode,
                    retrieved_pairs=pairs,
                    relevance=query.relevance,
                    latency_ms=latency,
                    original_embedding=orig_emb,
                    expanded_embedding=exp_emb,
                )
            )
        return results

    def _run_hybrid(
        self,
        query: Query,
        top_k: int,
        runs: int,
    ) -> list[QueryMetrics]:
        retriever = HybridRetriever(
            self.embedding_model,
            self.vector_store,
            self.bm25_store,
            rrf_k=self.config.hybrid.rrf_k,
        )

        pairs, q_emb = retriever.retrieve(query.text, top_k)
        latency = measure_latency_ms(
            lambda: retriever.retrieve(query.text, top_k), runs
        )

        return [
            build_query_metrics(
                query_id=query.id,
                query_text=query.text,
                strategy="hybrid",
                expansion_mode=None,
                retrieved_pairs=pairs,
                relevance=query.relevance,
                latency_ms=latency,
                original_embedding=None,
                expanded_embedding=None,
            )
        ]

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embedding_model(config: BenchmarkConfig) -> EmbeddingModel:
        from embeddings.local import LocalEmbeddingModel
        return LocalEmbeddingModel(config.embedding.model_name)

    @staticmethod
    def _build_vector_store(config: BenchmarkConfig) -> VectorStore:
        if config.vector_store.type == "numpy":
            from vector_store.numpy_store import NumpyVectorStore
            return NumpyVectorStore(config.embedding.dimension)
        from vector_store.faiss_store import FAISSVectorStore
        return FAISSVectorStore(config.embedding.dimension)
