"""Load WELFake dataset (Zenodo). Labels: 0=fake, 1=real — we invert to 0=credible, 1=misinformation."""
import csv
import os

MAX_TEXT_LEN = 1000
SOURCE_NAME = "welfake"


def load_welfake(data_dir=None):
    """Load WELFake from data/raw/welfake/. WELFake 0=fake -> label 1, 1=real -> label 0."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "welfake")
    path = os.path.join(data_dir, "WELFake.csv")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("text") or r.get("title") or "").strip()
            if not text:
                continue
            text = text[:MAX_TEXT_LEN]
            raw_label = r.get("label", "1")
            try:
                welfake_label = int(raw_label)
            except (ValueError, TypeError):
                welfake_label = 1
            # WELFake: 0=fake, 1=real -> our 0=credible, 1=misinformation
            our_label = 0 if welfake_label == 1 else 1
            rows.append({
                "text": text,
                "label": our_label,
                "category": "news",
                "source": SOURCE_NAME,
            })
    return rows
