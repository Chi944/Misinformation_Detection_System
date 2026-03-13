"""Load COVID Fake News dataset (HuggingFace / local CSV)."""
import csv
import os

MAX_TEXT_LEN = 1000
SOURCE_NAME = "covid"


def load_covid(data_dir=None):
    """Load COVID fake news from data/raw/covid/. Expects CSV with text and label (real/fake or 0/1)."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "covid")
    rows = []
    if not os.path.isdir(data_dir):
        return []
    for name in os.listdir(data_dir):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                text = (r.get("text") or r.get("content") or r.get("tweet") or "").strip()[:MAX_TEXT_LEN]
                if not text:
                    continue
                raw = (r.get("label") or r.get("label_quality") or "real").strip().lower()
                if raw in ("1", "fake", "false"):
                    label = 1
                else:
                    label = 0
                rows.append({
                    "text": text,
                    "label": label,
                    "category": "covid",
                    "source": SOURCE_NAME,
                })
    return rows


if __name__ == "__main__":
    rows = load_covid()
    print("%d rows" % len(rows))
