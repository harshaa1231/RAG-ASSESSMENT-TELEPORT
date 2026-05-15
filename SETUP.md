# RAG Benchmark — Setup Guide

> Read this first. You only need to do this once.

---

## What This Is

A working AI retrieval system that compares two search strategies on Google Cloud
documentation. It runs entirely on your laptop — no cloud account, no API keys needed.

When it finishes it produces three files:
- **`dashboard.html`** — interactive results dashboard, open in any browser (no internet needed)
- **`retrieval_benchmark.md`** — human-readable report (open in any text editor or VS Code)
- **`benchmark_results.json`** — raw metrics data

---

## Before You Start — Install Python

You need **Python 3.10 or newer**. Check if you already have it:

**Windows** — open Command Prompt and type:
```
python --version
```

**Mac** — open Terminal and type:
```
python3 --version
```

If it prints `Python 3.10.x` or higher, you are ready. Skip to the next section.

### Python Not Installed?

**Windows**
1. Go to **https://www.python.org/downloads/**
2. Click the yellow **"Download Python 3.x.x"** button
3. Run the installer
4. On the first screen, tick **"Add Python to PATH"** ← this is important
5. Click **Install Now**
6. After install, close and reopen Command Prompt, then run `python --version`

**Mac**
1. Open Terminal
2. Run: `brew install python3`
3. If you don't have Homebrew: go to **https://brew.sh** and follow the one-line install first

---

## Step 1 — Extract the ZIP

Unzip `rag-benchmark.zip` anywhere on your computer.
For example: `Desktop/rag-benchmark/`

---

## Step 2 — Run It

Pick whichever method suits you best — they all produce the same output.

---

### Option A: One-click scripts (simplest)

**Windows** — double-click `run.bat`, or from Command Prompt:
```cmd
cd Desktop\rag-benchmark
run.bat
```

**Mac / Linux** — from Terminal:
```bash
cd ~/Desktop/rag-benchmark
chmod +x run.sh && ./run.sh
```

> **First run downloads the AI model (~1.4 GB)** — takes a few minutes on first run.
> Every run after that is fast (~60 seconds).

When finished you will see:
```
[OK] benchmark_results.json
[OK] retrieval_benchmark.md
[OK] dashboard.html  ← open in any browser, no server needed
```

---

### Option B: Terminal step-by-step (Python 3.10+)

If you prefer to run each step manually from the terminal:

**Mac / Linux:**
```bash
# 1. Go to the project folder
cd ~/Desktop/rag-benchmark

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the benchmark
python -m benchmark.runner
```

**Windows (Command Prompt):**
```cmd
cd Desktop\rag-benchmark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m benchmark.runner
```

> Steps 2–4 only need to be done once. Next time, just activate the venv (step 3)
> and run the benchmark (step 5).

---

### Option C: Docker (any OS, no Python install needed)

```bash
docker compose build
docker compose run --rm rag-benchmark
```

---

## Step 3 — View the Results

After the run completes, three files appear in the project folder:

| File | How to open |
|------|-------------|
| `dashboard.html` | **Double-click** — opens in your browser instantly, no internet needed |
| `retrieval_benchmark.md` | VS Code, Notepad, TextEdit, or any text editor |
| `benchmark_results.json` | VS Code, Notepad, or any text editor |

**The dashboard is the easiest way to see results.** It shows charts comparing
Strategy A vs Strategy B across all 5 queries with colour-coded relevance scores.

To open the dashboard from the terminal:
```bash
# Mac
open dashboard.html

# Windows
start dashboard.html

# Linux
xdg-open dashboard.html
```

---

## Run the Tests (Optional)

To verify everything is working correctly:

**One-click:**
```bash
# Mac / Linux
./run.sh --test

# Windows
run.bat --test
```

**Or from terminal directly (venv must be activated):**
```bash
pytest tests/ -v
```

Expected result: `91 passed` in a few seconds (the tests use a fast fake model,
not the real 1.4 GB one).

---

## Troubleshooting

**"python is not recognised" on Windows**
> Python was installed without adding to PATH. Re-run the Python installer,
> select "Modify", tick "Add to PATH".

**First run is very slow**
> Normal — it is downloading a 1.4 GB AI model from the internet.
> Subsequent runs take ~60 seconds.

**`faiss-cpu` install fails on Mac (Apple Silicon)**
> Run: `pip install faiss-cpu --extra-index-url https://pypi.org/simple/`
> Or edit `config.yaml` and change `vector_store: type: faiss` to `type: numpy`.

**"Permission denied" on run.sh (Mac/Linux)**
> Run: `chmod +x run.sh` then `./run.sh`

**The window closes immediately (Windows)**
> Run from Command Prompt instead:
> 1. Open Command Prompt
> 2. `cd C:\path\to\rag-benchmark`
> 3. `run.bat`

---

## What Happens Behind the Scenes

```
run.bat / run.sh   (or: python -m benchmark.runner)
    │
    ├── Creates a Python virtual environment (.venv folder)
    ├── Installs all packages from requirements.txt
    ├── Downloads the AI embedding model (first run only)
    │
    └── python -m benchmark.runner
            ├── Loads 10 GCP technical paragraphs
            ├── Runs 5 test queries × 5 retrieval strategies
            ├── Measures MRR, NDCG, Precision, Latency, Cosine Drift
            ├── Writes benchmark_results.json
            ├── Writes retrieval_benchmark.md
            └── Writes dashboard.html  ← open this in your browser
```

Total runtime after first run: approximately **60–90 seconds**.

---

## Need More Detail?

- Full technical documentation: **README.md**
- Vertex AI production migration: **MIGRATION_GUIDE.md**
- Assessment context: see comments in `benchmark/corpus.py` and `benchmark/queries.py`
