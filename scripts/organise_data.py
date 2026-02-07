#!/usr/bin/env python3
"""Copy raw data from project root into data/raw. Run once to organise files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import organise_data_folders

if __name__ == "__main__":
    organise_data_folders()
    print("Done. Data is in data/raw/")
