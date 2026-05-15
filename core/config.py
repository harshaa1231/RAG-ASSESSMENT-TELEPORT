"""Pydantic-based settings loaded from config.yaml.

RAGEngine.from_config('config.yaml') is the canonical entry point.
No hardcoded paths — all file locations flow through OutputConfig.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class VectorStoreConfig(BaseModel):
    type: str = "faiss"  # "faiss" | "numpy"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    latency_runs: int = 5


class StrategyBConfig(BaseModel):
    default_mode: str = "good_expansion"


class HybridConfig(BaseModel):
    rrf_k: int = Field(default=60, ge=1)


class OutputConfig(BaseModel):
    results_json: str = "benchmark_results.json"
    report_md: str = "retrieval_benchmark.md"


class BenchmarkConfig(BaseModel):
    """Root configuration object for the RAG benchmark system."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    strategy_b: StrategyBConfig = Field(default_factory=StrategyBConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to config.yaml (relative or absolute).

        Returns:
            Populated BenchmarkConfig instance.
        """
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)
