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

