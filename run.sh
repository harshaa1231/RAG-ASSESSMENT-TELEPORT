#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  One-command setup and run for Linux / macOS
#
# Usage:
#   chmod +x run.sh
#   ./run.sh            # run benchmark
#   ./run.sh --test     # run pytest suite instead
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Check Python ──────────────────────────────────────────────────────────────
info "Checking Python version..."
$PYTHON --version || error "Python 3 not found. Install Python 3.10+ from https://python.org"

PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required. Found $($PYTHON --version)"
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing dependencies (first run downloads ~1.4 GB model)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed."

# ── Run ──────────────────────────────────────────────────────────────────────
echo ""
if [ "${1:-}" = "--test" ]; then
    info "Running pytest suite..."
    pytest tests/ -v
else
    info "Running RAG benchmark..."
    python -m benchmark.runner
    echo ""
    info "Output files generated:"
    echo "  benchmark_results.json"
    echo "  retrieval_benchmark.md"
fi
