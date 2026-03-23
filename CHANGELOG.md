# Changelog

All notable changes are documented here.

---

## [Current] - Phase 17 Complete

### Added
- Domain credibility scoring (80+ domains, data/domain_reputation.json)
- Explainability output (top words per model, explain=True parameter)
- LLM judge upgraded to mistral with llama3/llama2 fallback
- src/utils/domain_credibility.py
- src/utils/explainability.py

### Performance
- Ensemble accuracy: 0.682 (all-time best)
- BERT accuracy: 0.620 (retrained on mixed ISOT+LIAR+synthetic data)
- TF-IDF accuracy: 0.656
- NB accuracy: 0.634
- Ensemble weights: bert=0.1, tfidf=0.8, nb=0.1

---

## Phase 16 - Real Data Integration

### Added
- 80,000 real training samples (ISOT + LIAR + WELFake + COVID + FakeNewsNet)
- Grid search for optimal ensemble weights on validation set
- BERT fine-tuned on Kaggle GPU (Tesla T4)

---

## Phase 15 - CI/CD

### Added
- GitHub Actions workflow (.github/workflows/ci.yml)
- Dev container configuration (.devcontainer/)

---

## Phase 0-14 - Core System

### Added
- Three-model ensemble (BERT + TF-IDF DNN + Naive Bayes)
- Mamdani fuzzy logic engine (18 rules, Python 3.12 compatible)
- BackpropFeedbackLoop with EWC regularisation
- Local Ollama LLM judge (no API key required)
- Evaluation pipeline with dashboard
- 54 pytest tests
