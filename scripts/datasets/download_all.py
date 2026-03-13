"""Auto-download datasets where possible; print instructions for manual downloads (ISOT, FakeNewsNet)."""
import argparse
import csv
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "raw")
DATA_RAW = os.path.abspath(DATA_RAW)


def _load_hf_datasets():
    """Import HuggingFace datasets (avoid shadowing by this package named 'datasets')."""
    import site
    try:
        sites = [p for p in site.getsitepackages() if "site-packages" in p]
        if not sites:
            sites = site.getsitepackages()
        if sites:
            sys.path.insert(0, sites[0])
        try:
            from datasets import load_dataset as _load
            return _load
        finally:
            if sites:
                sys.path.pop(0)
    except Exception:
        return None


def download_liar():
    """Download LIAR from HuggingFace to data/raw/liar/ (train.tsv, valid.tsv, test.tsv)."""
    load_dataset = _load_hf_datasets()
    if load_dataset is None:
        print("Install datasets: pip install datasets", file=sys.stderr)
        return False
    out_dir = os.path.join(DATA_RAW, "liar")
    os.makedirs(out_dir, exist_ok=True)
    try:
        ds = load_dataset("ucsbnlp/liar")
    except Exception as e:
        print("LIAR download failed:", e, file=sys.stderr)
        return False
    # ucsbnlp/liar: 'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', ...
    for split_name, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        if hf_split not in ds:
            continue
        path = os.path.join(out_dir, "%s.tsv" % split_name)
        rows = ds[hf_split]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            for i, row in enumerate(rows):
                sid = row.get("id", i)
                label = row.get("label", 0)
                stmt = (row.get("statement") or "").strip()
                if not stmt:
                    continue
                w.writerow([sid, label, stmt])
        print("Wrote %s: %d rows" % (path, len(rows)))
    return True


def download_covid():
    """Download COVID fake news from HuggingFace to data/raw/covid/covid_fake_news.csv."""
    load_dataset = _load_hf_datasets()
    if load_dataset is None:
        print("Install datasets: pip install datasets", file=sys.stderr)
        return False
    out_dir = os.path.join(DATA_RAW, "covid")
    os.makedirs(out_dir, exist_ok=True)
    try:
        ds = load_dataset("nanyy1025/covid_fake_news")
    except Exception as e:
        print("COVID download failed:", e, file=sys.stderr)
        return False
    path = os.path.join(out_dir, "covid_fake_news.csv")
    # Common column names: tweet/text/content, label/label_quality (real/fake or 0/1)
    text_key = None
    label_key = None
    if "train" in ds and len(ds["train"]) > 0:
        first = ds["train"][0]
        for k in ("tweet", "text", "content", "Tweet", "Text"):
            if k in first:
                text_key = k
                break
        for k in ("label", "label_quality", "Label"):
            if k in first:
                label_key = k
                break
    if not text_key:
        text_key = "tweet"
    if not label_key:
        label_key = "label"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        for split in ("train", "test", "validation"):
            if split not in ds:
                continue
            for row in ds[split]:
                text = (row.get(text_key) or "").strip()
                if not text:
                    continue
                lb = row.get(label_key, "real")
                w.writerow({"text": text, "label": lb})
    total = sum(len(ds[s]) for s in ("train", "test", "validation") if s in ds)
    print("Wrote %s: %d rows" % (path, total))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-only", action="store_true", help="Only run auto-download (LIAR, COVID from HuggingFace)")
    args = ap.parse_args()
    os.makedirs(DATA_RAW, exist_ok=True)
    print("Data directory:", DATA_RAW)
    if args.auto_only:
        ok_liar = download_liar()
        ok_covid = download_covid()
        if ok_liar or ok_covid:
            print("Auto-download done.")
        else:
            print("No datasets downloaded. Install: pip install datasets", file=sys.stderr)
        return 0
    print()
    print("Auto-download: Use --auto-only to fetch LIAR and COVID from HuggingFace.")
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
