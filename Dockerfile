# ── Stage 1: dependency layer (cached unless requirements.txt changes) ─────
FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so every subsequent run is offline-capable.
# This adds ~1.4 GB to the image but eliminates the first-run wait.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"


# ── Stage 2: application ────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY . .

# Output directory for generated reports (mount a volume here to extract files)
RUN mkdir -p /output

ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "benchmark.runner"]
