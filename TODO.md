# Misinformation Detector — Build To-Do List

## PHASE 0: Audit & Setup
- [x] 0.1 Scan all existing files, remove dead code, unused imports, debug prints
- [x] 0.2 Consolidate shared utilities into src/utils/helpers.py
- [x] 0.3 Write requirements.txt with all pinned versions
- [x] 0.4 Create full folder structure with empty placeholder files

## PHASE 1: Models
- [x] 1.1 Write src/models/bert_classifier.py
- [x] 1.2 Write src/models/naive_bayes_model.py
- [x] 1.3 Write src/models/tfidf_model.py
- [x] 1.4 Write src/models/ensemble_detector.py
- [x] 1.5 Add AccuracyGateError and gate checks to src/training/trainer.py

## PHASE 2: Fuzzy Logic
- [x] 2.1 Write src/fuzzy/membership_functions.py
- [x] 2.2 Write src/fuzzy/fuzzy_engine.py

## PHASE 3: Feedback Loop
- [x] 3.1 Write src/feedback/feedback_store.py
- [x] 3.2 Write src/feedback/online_trainer.py
- [x] 3.3 Write src/feedback/backprop_loop.py

## PHASE 4: LLM Judge (Ollama — no API key)
- [x] 4.1 Write src/evaluation/llm_judge.py using Ollama local API

## PHASE 5: Evaluation
- [x] 5.1 Write src/evaluation/metrics.py
- [x] 5.2 Write src/evaluation/dashboard.py
- [x] 5.3 Write src/evaluation/pipeline.py

## PHASE 6: Master Class
- [x] 6.1 Write src/detector.py

## PHASE 7: Training Pipeline
- [x] 7.1 Write src/training/dataset.py
- [x] 7.2 Write src/training/calibration.py
- [x] 7.3 Write src/training/trainer.py
- [x] 7.4 Write scripts/train_all.py

## PHASE 8: CI/CD & Codespaces
- [x] 8.1 Write .github/workflows/ci.yml
- [x] 8.2 Write .devcontainer/devcontainer.json
- [x] 8.3 Write .devcontainer/postCreateCommand.sh (include ollama health check)

## PHASE 9: Git Automation
- [x] 9.1 Write src/utils/git_manager.py

## PHASE 10: Supporting Files
- [x] 10.1 Write Makefile
- [x] 10.2 Write scripts/smoke_test.py (include ollama health check)
- [x] 10.3 Write scripts/evaluate.py
- [x] 10.4 Write scripts/run_feedback_cycle.py
 - [x] 10.5 Write scripts/download_sample_data.py

## PHASE 11: Utilities
- [x] 11.1 Write src/utils/logger.py
- [x] 11.2 Write src/utils/gpu_utils.py
- [x] 11.3 Write src/utils/helpers.py

## PHASE 12: Tests
- [x] 12.1 Write tests/test_accuracy_gate.py
- [x] 12.2 Write tests/test_fuzzy_rules.py
- [x] 12.3 Write tests/test_feedback_loop.py
- [x] 12.4 Write tests/test_ensemble.py
- [x] 12.5 Write tests/test_llm_judge.py
- [x] 12.6 Write tests/test_bert.py
- [x] 12.7 Write tests/test_tfidf.py
- [x] 12.8 Write tests/test_naive_bayes.py

## PHASE 13: Configuration
- [x] 13.1 Write config.yaml (include llm_judge.host and llm_judge.model)
- [x] 13.2 Write .env.example
- [x] 13.3 Write .gitignore

## PHASE 14: Documentation
- [x] 14.1 Write README.md (include ollama setup instructions)
- [x] 14.2 Write ARCHITECTURE.md
- [x] 14.3 Write FEEDBACK_LOOP.md (include ollama pull instruction)
- [x] 14.4 Write CHANGELOG.md
- [x] 14.5 Write data/README.md
- [x] 14.6 Add full docstrings to every class and public method in src/

## PHASE 15: Final Execution
- [x] 15.1 Run: pip install -r requirements.txt
- [x] 15.2 Run: python scripts/download_sample_data.py
- [x] 15.3 Run: python scripts/train_all.py
- [x] 15.4 Run: python scripts/evaluate.py
- [x] 15.5 Run: python tests/test_accuracy_gate.py
- [x] 15.6 Run: python scripts/run_feedback_cycle.py
- [x] 15.7 Run: python scripts/evaluate.py (second pass post-feedback)
- [x] 15.8 Run: python scripts/smoke_test.py
- [x] 15.9 Run: pytest tests/ -v --asyncio-mode=auto
- [x] 15.10 Run: black src/ tests/ scripts/
- [x] 15.11 Run: flake8 src/ tests/ scripts/ --max-line-length=100
- [x] 15.12 Run: git add -A
- [x] 15.13 Run: git commit -m "feat: complete misinformation detector v1.0.0"
- [x] 15.14 Run: git push origin main

## PHASE 16: Real Data Integration
- [x] 16.1 Download ISOT dataset from Kaggle
- [x] 16.2 Download LIAR dataset from HuggingFace
- [x] 16.3 Download WELFake dataset from Zenodo
- [x] 16.4 Download COVID Fake News dataset from HuggingFace
- [x] 16.5 Download FakeNewsNet dataset from GitHub
- [x] 16.6 Run: python scripts/datasets/download_all.py
- [x] 16.7 Run: python scripts/combine_datasets.py
- [x] 16.8 Run: python scripts/train_all.py --data data/train.csv
- [x] 16.9 Run: python scripts/evaluate.py (verify gates pass on real data)
- [x] 16.10 Run: python scripts/run_feedback_cycle.py

## PHASE 17: Optional Enhancements
- [ ] 17.1 Add a REST API endpoint using FastAPI
- [ ] 17.2 Add a simple web UI for live predictions
- [ ] 17.3 Add multilingual support (non-English misinformation)
- [ ] 17.4 Add source credibility database (known reliable/unreliable domains)
- [ ] 17.5 Add explainability output (which words drove the prediction)
- [ ] 17.6 Connect Ollama to a larger model (llama3:70b or mistral)
- [ ] 17.7 Deploy to a cloud server or HuggingFace Spaces
