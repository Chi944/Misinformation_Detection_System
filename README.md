# Misinformation Detection System

A production-grade misinformation detection system combining three machine
learning models, fuzzy logic, a local LLM judge, and a feedback loop.

**GitHub:** https://github.com/Chi944/Misinformation_Detection_System

---

## Architecture

| Component | Technology | Role |
|---|---|---|
| BERT | PyTorch, bert-base-uncased | Transformer-based text classifier |
| TF-IDF DNN | TensorFlow/Keras, word+char n-grams | Deep neural text classifier |
| Naive Bayes | scikit-learn, ComplementNB | Fast probabilistic baseline |
| Ensemble | Weighted combination | Final prediction (grid-search weights) |
| Fuzzy Logic | Mamdani, 18 rules | Confidence calibration |
| LLM Judge | Ollama (mistral) | Optional natural language reasoning |
| Feedback Loop | EWC regularisation | Online learning from corrections |

---

## Model Performance

Evaluated on 500 held-out test samples from ISOT, LIAR and WELFake datasets.
Ensemble weights optimised via grid search on validation set.

| Model | Accuracy | F1 | Precision | Weight |
|---|---|---|---|---|
| BERT | 0.620 | 0.632 | 0.542 | 10% |
| TF-IDF DNN | 0.656 | 0.653 | 0.616 | 80% |
| Naive Bayes | 0.634 | 0.623 | 0.627 | 10% |
| **Ensemble** | **0.682** | **0.663** | **0.644** | - |

- 0 prediction errors on 500-sample evaluation
- 54 pytest tests passing, 8/8 smoke tests passing
- BERT fine-tuned on mixed ISOT + LIAR + synthetic data (Google Colab T4 GPU)

---

## Phase 17 Enhancements

| Feature | Status | Description |
|---|---|---|
| Domain credibility | Active | Scores 80+ known domains, adjusts ensemble probability |
| Explainability | Active | Top words driving each prediction (TF-IDF + NB) |
| LLM judge model | Active | Using mistral via Ollama (falls back to llama3/llama2) |

---

## Quick Start

See [RUN.md](RUN.md) for detailed step-by-step instructions.

```bash
# 1. Clone
git clone https://github.com/Chi944/Misinformation_Detection_System.git
cd Misinformation_Detection_System

# 2. Install
pip install -r requirements.txt

# 3. Verify everything works
python scripts/smoke_test.py --synthetic

# 4. Run a prediction
python -c "from src.detector import MisinformationDetector; d = MisinformationDetector(config='config.yaml'); r = d.predict('SHOCKING cover-up exposed by insiders!'); print(r['crisp_label'], r['ensemble_probability'])"
```

---

## Project Structure

```
src/
  models/       bert_classifier.py, tfidf_model.py,
                naive_bayes_model.py, ensemble_detector.py
  fuzzy/        fuzzy_engine.py, membership_functions.py
  feedback/     backprop_loop.py, online_trainer.py, feedback_store.py
  evaluation/   llm_judge.py, metrics.py, dashboard.py, pipeline.py
  training/     dataset.py, trainer.py, calibration.py
  utils/        logger.py, gpu_utils.py, helpers.py, skfuzzy_compat.py,
                domain_credibility.py, explainability.py
scripts/
  smoke_test.py, evaluate.py, train_all.py,
  run_feedback_cycle.py, combine_datasets.py
tests/          54 passing, 5 skipped
data/           train.csv (80k), val.csv (10k), test.csv (10k)
models/         bert_classifier.pt, tfidf_model.keras,
                tfidf_vectorizer.joblib, naive_bayes.pkl, nb_vectorizer.pkl
```

---

## Ensemble Weights

Current weights (bert=0.1, tfidf=0.8, nb=0.1) were found via grid search
on the validation set. To re-optimise after retraining:

```bash
python scripts/grid_search_weights.py
```

---

## LLM Judge (Optional)

The LLM judge adds natural language reasoning to predictions.
It is optional — the ensemble works without it.

```bash
ollama serve
ollama pull mistral
curl http://localhost:11434/api/tags
```

---

## Retraining

The pre-trained models are ready to use. Only retrain if you have new data.

```bash
# Retrain NB + TF-IDF only (20 min, no GPU needed)
python scripts/train_all.py --data data/train.csv --skip-bert --skip-gates
```

BERT retraining requires a GPU. See the Colab notebook in docs/.

---

## Test Results

```
54 passed, 5 skipped, 0 failed
Smoke test: 8/8 passed
```

Skipped tests require Ollama to be running or a GPU.

---

## Notes

- Python 3.12 required
- Fuzzy engine uses manual Mamdani inference (no scikit-fuzzy ControlSystem)
  via src/utils/skfuzzy_compat.py for Python 3.12 compatibility
- All model weights are local — no external API keys required
- LLM judge uses local Ollama only
