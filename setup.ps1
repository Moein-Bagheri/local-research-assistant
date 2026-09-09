$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}

.\.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
ollama pull all-minilm
.\.venv\Scripts\python scripts\build_index.py
.\.venv\Scripts\python scripts\generate_demo_outputs.py

Write-Host "Setup complete. Run: .\.venv\Scripts\python cli.py"

