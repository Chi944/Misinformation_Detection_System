"""Auto-download datasets where possible; print instructions for manual downloads (ISOT, FakeNewsNet)."""
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "raw")
DATA_RAW = os.path.abspath(DATA_RAW)


def main():
    os.makedirs(DATA_RAW, exist_ok=True)
    print("Data directory:", DATA_RAW)
    print()
    print("Auto-download: Not all datasets can be fetched without API keys or login.")
    print("  - LIAR: download from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
    print("  - WELFake: download from Zenodo (link in dataset page)")
    print("  - COVID: use HuggingFace datasets or place CSV in data/raw/covid/")
    print()
    print("Manual steps required:")
    print("  1. ISOT: Download from Kaggle (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
    print("     Extract True.csv and Fake.csv into data/raw/isot/")
    print("  2. FakeNewsNet: Clone or download from GitHub")
    print("     https://github.com/KaiDMML/FakeNewsNet")
    print("     Place processed CSV/JSON in data/raw/fakenewsnet/")
    print()
    print("Then run: python scripts/combine_datasets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
