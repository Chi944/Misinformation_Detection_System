#!/usr/bin/env python3
"""Verify the app compiles and runs: imports, data load, train, API health."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def step(name):
    print(f"[verify] {name} ...")

step("Importing config")
from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
assert PROJECT_ROOT.is_dir(), "PROJECT_ROOT missing"

step("Importing data_preprocessing")
from src.data_preprocessing import organise_data_folders, prepare_data, DatasetLoader

step("Organising data folders")
organise_data_folders()

step("Preparing data (synthetic fallback)")
train_df, val_df, test_df = prepare_data(use_synthetic=True, use_hf=False, use_fakenewsnet=False)
assert len(train_df) > 0 and "combined_text" in train_df.columns
print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

step("Importing traditional_ml and training")
from src.traditional_ml import train_single_model
train_single_model(train_df, val_df)

step("Loading inference and predicting")
from src.inference import load_models, InferenceEngine
load_models()
engine = InferenceEngine()
out = engine.predict("Breaking: This is a test headline!", url=None)
assert "prediction" in out and "credibility_audit" in out
print(f"  prediction={out['prediction']}, model={out['model']}")

step("All checks passed")
print("[verify] App compiles and runs successfully.")
