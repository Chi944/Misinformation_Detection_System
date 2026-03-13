"""Combine all available dataset loaders into a single CSV (text, label, category, source)."""
import argparse
import csv
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from scripts.datasets.load_isot import load_isot
from scripts.datasets.load_liar import load_liar
from scripts.datasets.load_welfake import load_welfake
from scripts.datasets.load_covid import load_covid
from scripts.datasets.load_fakenewsnet import load_fakenewsnet

LOADERS = {
    "isot": load_isot,
    "liar": load_liar,
    "welfake": load_welfake,
    "covid": load_covid,
    "fakenewsnet": load_fakenewsnet,
}


def main():
    ap = argparse.ArgumentParser(description="Combine dataset loaders into one CSV")
    ap.add_argument("--sources", nargs="+", default=None, help="Use only these sources (e.g. isot liar)")
    ap.add_argument("--max-per-source", type=int, default=None, help="Cap rows per source")
    ap.add_argument("--max-per-class", type=int, default=None, help="Cap rows per label (0/1)")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: data/train.csv)")
    args = ap.parse_args()
    out_path = args.output or os.path.join(_root, "data", "train.csv")
    sources = [s.lower() for s in (args.sources or list(LOADERS.keys()))]
    all_rows = []
    for name in sources:
        if name not in LOADERS:
            print("Unknown source:", name, file=sys.stderr)
            continue
        try:
            rows = LOADERS[name]()
        except Exception as e:
            print("Error loading %s: %s" % (name, e), file=sys.stderr)
            continue
        if args.max_per_source and len(rows) > args.max_per_source:
            rows = rows[: args.max_per_source]
        all_rows.extend(rows)
    if args.max_per_class is not None:
        by_label = {0: [], 1: []}
        for r in all_rows:
            by_label[r["label"]].append(r)
        all_rows = []
        for k in (0, 1):
            all_rows.extend(by_label[k][: args.max_per_class])
    if not all_rows:
        print("No data found. Download datasets and place in data/raw/<source>/ as per scripts/datasets/download_all.py", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category", "source"])
        w.writeheader()
        w.writerows(all_rows)
    print("Wrote %d rows to %s" % (len(all_rows), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
