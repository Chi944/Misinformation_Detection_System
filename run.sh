#!/usr/bin/env bash
# Run app in GitHub Codespaces or any Linux environment.
set -e
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

case "${1:-}" in
  train)
    echo "Training models..."
    python main.py --train
    ;;
  api)
    echo "Starting API on port 5000..."
    python main.py --api --port 5000
    ;;
  demo)
    python main.py --demo
    ;;
  *)
    echo "Usage: ./run.sh {train|api|demo}"
    echo "  train - train models (run once)"
    echo "  api   - start Flask API"
    echo "  demo  - CLI demo"
    exit 1
    ;;
esac
