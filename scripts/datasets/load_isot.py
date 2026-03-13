"""Load ISOT fake news dataset (Kaggle). Labels: 0=real, 1=fake."""
import csv
import os

MAX_TEXT_LEN = 1000
SOURCE_NAME = "isot"


def load_isot(data_dir=None):
    """Load ISOT dataset from data/raw/isot/. Returns list of dicts: text, label, category, source."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "isot")
    rows = []
    for name in ("True.csv", "Fake.csv"):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        label = 1 if "Fake" in name else 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                text = (r.get("text") or r.get("title") or "").strip()
                if not text and "title" in r:
                    text = (r.get("title") or "").strip()
                text = (text or "")[:MAX_TEXT_LEN]
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "label": label,
                    "category": "news",
                    "source": SOURCE_NAME,
                })
    return rows
