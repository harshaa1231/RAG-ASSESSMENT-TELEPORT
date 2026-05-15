.PHONY: install run test clean docker-build docker-run docker-test help

PYTHON ?= python3
VENV   = .venv
PIP    = $(VENV)/bin/pip

# ── Local targets ────────────────────────────────────────────────────────────

install:           ## Install all dependencies into a local virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:               ## Run the full benchmark (generates benchmark_results.json + retrieval_benchmark.md)
	$(VENV)/bin/python -m benchmark.runner

test:              ## Run pytest suite (91 tests)
	$(VENV)/bin/pytest tests/ -v

clean:             ## Remove generated files and caches
	rm -rf $(VENV) __pycache__ .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f benchmark_results.json retrieval_benchmark.md

# ── Docker targets ───────────────────────────────────────────────────────────

docker-build:      ## Build the Docker image (downloads model, ~1.5 GB)
	docker compose build

docker-run:        ## Run benchmark inside Docker (output files appear in current dir)
	docker compose run --rm rag-benchmark

docker-test:       ## Run pytest suite inside Docker
	docker compose run --rm rag-test

# ── Help ─────────────────────────────────────────────────────────────────────

help:              ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN{FS=":.*##"}{printf "  %-18s %s\n",$$1,$$2}'
