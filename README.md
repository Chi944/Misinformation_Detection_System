# Misinformation Detection System

A production-grade misinformation detection system combining three machine
learning models, a Mamdani fuzzy logic engine, a local LLM judge, and a
backward propagation feedback loop with EWC regularisation.

**GitHub:** https://github.com/Chi944/Misinformation_Detection_System

## Architecture

- **BERT** (PyTorch) — transformer-based classifier (bert-base-uncased)
- **TF-IDF DNN** (TensorFlow/Keras) — word + char n-gram deep network
- **Naive Bayes** (scikit-learn) — fast probabilistic baseline with online learning
- **Ensemble** — weighted combination (BERT 50%, TF-IDF 30%, NB 20%)
- **Fuzzy Logic** — manual Mamdani inference engine, 18 rules, Python 3.12 safe
- **LLM Judge** — local Ollama (llama3), no API key required
- **Feedback Loop** — backward propagation with EWC regularisation

## Requirements

- Python 3.12
- Ollama for the LLM judge (optional — system works without it)

## Quick Start
`ash
# Clone and install
git clone https://github.com/Chi944/Misinformation_Detection_System.git
cd Misinformation_Detection_System
pip install -r requirements.txt

# Set up Ollama (optional but recommended)
ollama serve
ollama pull llama3

# Generate sample data
python scripts/download_sample_data.py

# Run smoke test to verify everything works
python scripts/smoke_test.py --synthetic

# Train all models on synthetic data
python scripts/train_all.py --synthetic --n-synthetic 500

# Evaluate
python scripts/evaluate.py --synthetic
`

## Ollama Setup

The LLM judge requires a locally running Ollama instance. All other
features work without it.
`ash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve

# Pull the model
ollama pull llama3

# Verify
curl http://localhost:11434/api/tags
`

## Makefile Commands

| Command | Description |
|---|---|
| make install | Install all dependencies |
| make smoke | Run smoke test (fast pipeline check) |
| make train | Train on real data |
| make train-syn | Train on 500 synthetic samples |
| make evaluate | Run full evaluation pipeline |
| make evaluate-syn | Evaluate on synthetic data |
| make feedback | Run one feedback cycle |
| make test | Run all pytest tests |
| make lint | Run flake8 linter |
| make format | Run black formatter |
| make clean | Remove generated files |

## Accuracy Gates

Training enforces minimum thresholds. Training fails if any model
falls below its gate:

| Model | Accuracy | Precision | F1 |
|---|---|---|---|
| BERT | 0.78 | 0.76 | 0.77 |
| TF-IDF | 0.76 | 0.75 | 0.75 |
| Naive Bayes | 0.75 | 0.75 | 0.74 |
| Ensemble | 0.82 | 0.80 | 0.81 |

## Project Structure
src/
models/         bert_classifier.py, tfidf_model.py,
naive_bayes_model.py, ensemble_detector.py
fuzzy/          fuzzy_engine.py, membership_functions.py
feedback/       backprop_loop.py, online_trainer.py, feedback_store.py
evaluation/     llm_judge.py, metrics.py, dashboard.py, pipeline.py
training/       dataset.py, trainer.py, calibration.py
utils/          logger.py, gpu_utils.py, helpers.py,
git_manager.py, skfuzzy_compat.py
scripts/
train_all.py, evaluate.py, smoke_test.py,
run_feedback_cycle.py, download_sample_data.py
tests/            8 pytest test files (53 passed, 6 skipped, 0 failed)
.github/          CI/CD GitHub Actions workflow
.devcontainer/    GitHub Codespaces configuration
data/             train.csv (800), val.csv (100), test.csv (100)
reports/          evaluation_report.json, evaluation_dashboard.png

## Test Results
53 passed, 6 skipped, 0 failed
Skipped tests require Ollama to be running or TensorFlow GPU.

## Notes

- .venv/ and node_modules/ are excluded from git
- The fuzzy engine uses manual Mamdani inference (no ControlSystem)
  via src/utils/skfuzzy_compat.py for Python 3.12 compatibility
- All LLM judge calls use local Ollama — no external API keys needed
