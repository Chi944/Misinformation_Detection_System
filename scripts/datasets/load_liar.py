"""Load LIAR dataset (PolitiFact). 6-way labels converted to binary: true=0, false=1."""
import csv
import os

MAX_TEXT_LEN = 1000
SOURCE_NAME = "liar"

# LIAR 6-way: 0=true, 1=mostly-true, 2=half-true, 3=mostly-false, 4=false, 5=pants-fire
# Map to binary: 0,1,2 -> credible (0); 3,4,5 -> misinformation (1)
def _to_binary(score):
    try:
        s = int(score)
        return 0 if s <= 2 else 1
    except (ValueError, TypeError):
        return 0


def load_liar(data_dir=None):
    """Load LIAR train.tsv, valid.tsv, test.tsv from data/raw/liar/ or data/raw/. Returns list of dicts."""
    if data_dir is None:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
        for candidate in [os.path.join(base, "liar"), base]:
            if os.path.isfile(os.path.join(candidate, "train.tsv")):
                data_dir = candidate
                break
        else:
            data_dir = os.path.join(base, "liar")
    rows = []
    for name in ("train.tsv", "valid.tsv", "test.tsv"):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for r in reader:
                if len(r) < 3:
                    continue
                # LIAR: id, label, statement, subject, speaker, ...
                label_idx = 1
                text_idx = 2
                text = (r[text_idx] if len(r) > text_idx else "").strip()[:MAX_TEXT_LEN]
                if not text:
                    continue
                label = _to_binary(r[label_idx])
                rows.append({
                    "text": text,
                    "label": label,
                    "category": "politics",
                    "source": SOURCE_NAME,
                })
    return rows


if __name__ == "__main__":
    rows = load_liar()
    print("%d rows" % len(rows))
