# Changelog

## [1.0.0] - 2025-01-01

### Added — Phase 0: Project Setup
- Project structure with all src/ subdirectories
- requirements.txt with Python 3.12 compatible versions
- All src/__init__.py files

### Added — Phase 1: Models
- BERTMisinformationClassifier (bert-base-uncased, PyTorch, dropout=0.3)
- TFIDFModel (word 50k + char 30k TF-IDF, Keras DNN)
- TFNaiveBayesWrapper (ComplementNB, online learning via partial_fit)
- EnsembleDetector (weighted average, BERT 50% + TF-IDF 30% + NB 20%)
- MasterTrainer with AccuracyGateError and per-model accuracy gates
  (BERT acc>=0.78, TF-IDF acc>=0.76, NB acc>=0.75, Ensemble acc>=0.82)

### Added — Phase 2: Fuzzy Logic
- FuzzyMisinformationEngine: manual Mamdani inference, 18 rules
- 6 antecedents, 1 consequent, centroid defuzzification
- skfuzzy_compat.py: patches removed imp module for Python 3.12
- No ControlSystem usage (crashes on Python 3.12)

### Added — Phase 3: Feedback Loop
- FeedbackStore: SQLite persistence, TF-IDF cosine similarity lookup
- OnlineTrainer: BERT (AdamW + EWC), TF-IDF (Keras fit), NB (partial_fit)
- BackpropFeedbackLoop: 9-step run_cycle() with git commit integration

### Added — Phase 4: LLM Judge
- LLMJudge: local Ollama (llama3), no API key required
- Endpoint: http://localhost:11434/api/generate
- FALLBACK_VERDICT for when Ollama is not running
- evaluate_single(), evaluate_batch(), generate_model_report()

### Added — Phase 5: Evaluation
- MetricsCalculator: accuracy, precision, recall, F1, ROC-AUC, ECE
- EvaluationDashboard: 6-panel PNG (confusion matrices, ROC curves,
  fuzzy membership, judge agreement, calibration, heatmap)
- EvaluationPipeline: orchestrates metrics + judge + dashboard

### Added — Phase 6: Master Detector
- MisinformationDetector: master class coordinating all components
- predict(), evaluate(), evaluate_quick() methods
- fast_mode for CI/smoke tests (skips heavy model downloads)

### Added — Phase 7: Training Pipeline
- MisinformationDataset: CSV/JSON loader, stratified split
- create_synthetic(): 200-sample keyword-based synthetic dataset
- TemperatureScaler and EnsembleCalibrator: post-hoc calibration
- scripts/train_all.py: CLI with --synthetic and --skip-gates flags

### Added — Phase 8: CI/CD
- .github/workflows/ci.yml: GitHub Actions, Python 3.12, Ollama install
- .devcontainer/devcontainer.json: GitHub Codespaces configuration
- .devcontainer/postCreateCommand.sh: auto-setup with Ollama pull

### Added — Phase 9: Git Automation
- GitManager: automated commit of feedback cycle results

### Added — Phase 10: Scripts
- scripts/smoke_test.py: 8-check pipeline verification
- scripts/evaluate.py: full evaluation with report output
- scripts/run_feedback_cycle.py: single feedback cycle runner
- scripts/download_sample_data.py: generates 800/100/100 CSV splits

### Added — Phase 11: Utilities
- get_logger(): cached logger with stdout + optional file handler
- get_device(), get_tf_device(), set_memory_growth(), log_device_info()
- clean_text(), safe_divide(), clamp(), hash_text(), label_to_str()

### Added — Phase 12: Tests
- 8 pytest test files: 53 passed, 6 skipped, 0 failed
- test_accuracy_gate, test_fuzzy_rules, test_feedback_loop,
  test_ensemble, test_llm_judge, test_bert, test_tfidf, test_naive_bayes

### Added — Phase 13: Configuration
- config.yaml: all model, training, gate, fuzzy, feedback, eval settings
- .env.example: environment variable template
- .gitignore: excludes .venv, node_modules, checkpoints, feedback.db

### Added — Phase 14: Documentation
- README.md, ARCHITECTURE.md, FEEDBACK_LOOP.md, CHANGELOG.md
- data/README.md with CSV format and data source references

### Added — Phase 15: Final Execution
- All 15 phases verified end-to-end
- Git history cleaned (removed .venv and node_modules via filter-repo)
- Final commit: 56bf361
- Repository: https://github.com/Chi944/Misinformation_Detection_System

### Technical Notes
- Python 3.12.7 on Windows
- scikit-fuzzy requires skfuzzy_compat.py (patches removed imp module)
- All print statements use %% formatting (Windows cp1252 safe)
- Lazy imports throughout to prevent import-time hangs
- Local Ollama — zero external API dependencies
