"""Load FakeNewsNet dataset (GitHub). Expects JSON or CSV in data/raw/fakenewsnet/."""
import csv
import json
import os

MAX_TEXT_LEN = 1000
SOURCE_NAME = "fakenewsnet"


def load_fakenewsnet(data_dir=None):
    """Load FakeNewsNet from data/raw/fakenewsnet/. Supports .json or .csv with text and label."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "fakenewsnet")
    if not os.path.isdir(data_dir):
        return []
    rows = []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".json"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, list):
                    for item in data:
                        text = (item.get("text") or item.get("title") or item.get("content") or "").strip()[:MAX_TEXT_LEN]
                        if not text:
                            continue
                        label = 1 if (item.get("label") in (1, "fake", "false") or item.get("fake") is True) else 0
                        rows.append({"text": text, "label": label, "category": "news", "source": SOURCE_NAME})
                elif isinstance(data, dict):
                    text = (data.get("text") or data.get("title") or data.get("content") or "").strip()[:MAX_TEXT_LEN]
                    if text:
                        label = 1 if (data.get("label") in (1, "fake", "false") or data.get("fake") is True) else 0
                        rows.append({"text": text, "label": label, "category": "news", "source": SOURCE_NAME})
        elif name.endswith(".csv"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    text = (r.get("text") or r.get("title") or r.get("news_content") or "").strip()[:MAX_TEXT_LEN]
                    if not text:
                        continue
                    raw = (r.get("label") or r.get("label_quality") or "real").strip().lower()
                    label = 1 if raw in ("1", "fake", "false") else 0
                    rows.append({"text": text, "label": label, "category": "news", "source": SOURCE_NAME})
    return rows
