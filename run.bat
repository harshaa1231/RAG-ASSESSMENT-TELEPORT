@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: run.bat  —  One-command setup and run for Windows
::
:: Usage:
::   run.bat             (run benchmark)
::   run.bat --test      (run pytest suite instead)
:: ─────────────────────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

set VENV_DIR=.venv
set PYTHON=python

echo.
echo ============================================================
echo  RAG Benchmark  ^|  Setup and Run
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PYTHON% --version 2^>^&1') do set PY_VER=%%v
echo [INFO]  Found Python %PY_VER%

:: ── Virtual environment ───────────────────────────────────────────────────────
if not exist "%VENV_DIR%\" (
    echo [INFO]  Creating virtual environment in %VENV_DIR%...
    %PYTHON% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
)

call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause & exit /b 1
)

:: ── Install dependencies ──────────────────────────────────────────────────────
echo [INFO]  Installing dependencies...
echo [INFO]  First run downloads the embedding model (~1.4 GB) — this may take a few minutes.
echo.
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause & exit /b 1
)
echo [INFO]  Dependencies ready.
echo.

:: ── Run ──────────────────────────────────────────────────────────────────────
if "%1"=="--test" (
    echo [INFO]  Running pytest suite...
    echo.
    pytest tests/ -v
) else (
    echo [INFO]  Running RAG benchmark...
    echo.
    python -m benchmark.runner
    echo.
    echo [INFO]  Output files generated:
    echo           benchmark_results.json
    echo           retrieval_benchmark.md
)

echo.
pause
