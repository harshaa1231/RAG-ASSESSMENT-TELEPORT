# RAG Benchmark — Setup Guide

> Read this first. You only need to do this once.

---

## What This Is

A working AI retrieval system that compares two search strategies on Google Cloud
documentation. It runs entirely on your laptop — no cloud account, no API keys needed.

When it finishes it produces two files:
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

### On Windows

1. Open the extracted `rag-benchmark` folder
2. **Double-click `run.bat`**
3. A window opens — this is normal. Let it run.
4. **First run downloads the AI model (~1.4 GB)** — this takes a few minutes depending
   on your internet speed. Every run after that is fast.
5. When done, the window shows `[OK] benchmark_results.json` and `[OK] retrieval_benchmark.md`

> If Windows says "Windows protected your PC", click **More info** → **Run anyway**.
> This happens because the script is not code-signed.

### On Mac / Linux

1. Open Terminal
2. Navigate to the extracted folder:
   ```
   cd ~/Desktop/rag-benchmark
   ```
3. Run:
   ```
   chmod +x run.sh && ./run.sh
   ```
4. **First run downloads the AI model (~1.4 GB)** — a few minutes on first run.
5. When done you'll see `[OK] benchmark_results.json` and `[OK] retrieval_benchmark.md`

---

## Step 3 — View the Results

After the run completes, open these files in the same folder:

| File | How to open |
|------|-------------|
| `retrieval_benchmark.md` | VS Code, Notepad, TextEdit, or any text editor |
| `benchmark_results.json` | VS Code, Notepad, or any text editor |

The markdown file is the main report. It shows how Strategy A (raw search) compares
to Strategy B (AI-enhanced search) across 5 queries.

---

## Run the Tests (Optional)

To verify everything is working correctly:

**Windows:**
```
run.bat --test
```

**Mac / Linux:**
```
./run.sh --test
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
run.bat / run.sh
    │
    ├── Creates a Python virtual environment (.venv folder)
    ├── Installs all packages from requirements.txt
    ├── Downloads the AI embedding model (first run only)
    │
    └── python -m benchmark.runner
            ├── Loads 10 GCP technical paragraphs
            ├── Runs 5 test queries × 5 retrieval strategies
            ├── Measures MRR, NDCG, Precision, Latency, Cosine Drift
            └── Writes benchmark_results.json + retrieval_benchmark.md
```

Total runtime after first run: approximately **60–90 seconds**.

---

## Need More Detail?

- Full technical documentation: **README.md**
- Vertex AI production migration: **MIGRATION_GUIDE.md**
- Assessment context: see comments in `benchmark/corpus.py` and `benchmark/queries.py`
