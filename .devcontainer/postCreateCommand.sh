#!/usr/bin/env bash
set -e

echo "=== Misinformation Detector Dev Container Setup ==="

echo "--- Installing Python dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "--- Installing Ollama ---"
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed"
else
    echo "Ollama already installed"
fi

echo "--- Starting Ollama server ---"
ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"
sleep 8

echo "--- Pulling llama3 model ---"
if ollama pull llama3; then
    echo "llama3 pulled successfully"
else
    echo "WARNING: llama3 pull failed - LLM judge will use fallback mode"
fi

echo "--- Verifying Ollama health ---"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama health check PASS"
else
    echo "WARNING: Ollama not responding on port 11434"
fi

echo "--- Creating required directories ---"
mkdir -p data reports checkpoints logs

echo "--- Downloading sample data (if available) ---"
python scripts/download_sample_data.py || echo "Sample data download skipped"

echo "=== Dev container setup complete ==="
echo "Run 'make help' to see available commands"
echo "Run 'make smoke' to verify the full pipeline"

#!/bin/bash
set -e
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p models reports data
if [ ! -f data/sample_train.csv ]; then python scripts/download_sample_data.py; fi
python scripts/smoke_test.py || true
OLLAMA_HOST=${LLM_JUDGE_HOST:-http://localhost:11434}
if curl -sSf "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is reachable at ${OLLAMA_HOST}."
else
    echo "WARNING: Ollama not reachable at ${OLLAMA_HOST}. LLM judge will be disabled."
fi
echo "Setup complete. Run: make train"

