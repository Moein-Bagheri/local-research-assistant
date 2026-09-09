$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Virtual environment is missing. Run .\setup.ps1 first."
}

if (-not (Test-Path -LiteralPath "artifacts\faiss\papers.index")) {
    .\.venv\Scripts\python scripts\build_index.py
}

.\.venv\Scripts\python cli.py

