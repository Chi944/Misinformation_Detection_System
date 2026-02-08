"""
Configuration settings for the misinformation detection system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model settings
RANDOM_SEED = 42

# Traditional ML settings (tuned for higher accuracy)
TFIDF_MAX_FEATURES = 25000
NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
LR_MAX_ITER = 2000

# Evaluation settings
TEST_SIZE = 0.1
VAL_SIZE = 0.1
TRAIN_SIZE = 0.8

# Inference constraints
MAX_INFERENCE_LATENCY_MS = 500

# Labels
LABEL_CREDIBLE = 0
LABEL_MISINFORMATION = 1
LABEL_NAMES = {0: "Credible", 1: "Misinformation"}
