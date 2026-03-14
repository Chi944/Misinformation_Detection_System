"""Check label balance in train/val/test CSVs to diagnose class-imbalance collapse.

Usage:
  python scripts/check_label_balance.py [data/train.csv [data/val.csv [data/test.csv]]]
Defaults: data/train.csv data/val.csv data/test.csv

If credible (0) >> misinfo (1), BERT may collapse to predicting 0 for everything.
Fix: retrain with class weights or oversample the minority class.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def check_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "label" not in rows[0]:
        return {"error": "no 'label' column or empty file"}
    labels = [int(r["label"]) for r in rows]
    c = Counter(labels)
    n0, n1 = c.get(0, 0), c.get(1, 0)
    total = len(labels)
    return {
        "total": total,
        "n0": n0,
        "n1": n1,
        "pct0": 100 * n0 / total if total else 0,
        "pct1": 100 * n1 / total if total else 0,
        "ratio_1_to_0": n1 / max(1, n0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Check label balance in CSV datasets.")
    p.add_argument(
        "files",
        nargs="*",
        default=["data/train.csv", "data/val.csv", "data/test.csv"],
        help="CSV files with a 'label' column (0=credible, 1=misinfo)",
    )
    args = p.parse_args()

    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f"{path}: file not found")
            continue
        name = path.name
        out = check_file(path)
        if "error" in out:
            print(f"{name}: {out['error']}")
            continue
        total = out["total"]
        n0, n1 = out["n0"], out["n1"]
        p0, p1 = out["pct0"], out["pct1"]
        r = out["ratio_1_to_0"]
        print(f"{name}: total={total}  0 (credible)={n0} ({p0:.1f}%)  1 (misinfo)={n1} ({p1:.1f}%)  ratio 1:0={r:.2f}")
        if p0 > 80 or p1 > 80:
            print(f"  -> WARNING: severe imbalance; use class weights or oversampling when training BERT.")

    return None


if __name__ == "__main__":
    main()
    sys.exit(0)
