"""Evaluate ensemble and base models on first 500 rows of data/test.csv."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import accuracy_score, f1_score, precision_score

from src.detector import MisinformationDetector


def main():
    detector = MisinformationDetector(config="config.yaml", fast_mode=False)
    w = detector.ensemble._get_active_weights()
    print("Weights:", {k: round(v, 2) for k, v in w.items()})

    with open("data/test.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:500]
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    print("Evaluating 500 samples...")
    ens_preds = []
    model_preds = {"bert": [], "tfidf": [], "naive_bayes": []}
    errors = 0

    for i, t in enumerate(texts):
        try:
            r = detector.predict(t)
            ep = 1 if r["ensemble_probability"] >= 0.5 else 0
            ens_preds.append(ep)
            bd = r["model_breakdown"]
            for m in model_preds:
                p = 1 if bd[m]["confidence"] >= 0.5 else 0
                model_preds[m].append(p)
        except Exception:
            errors += 1
            ens_preds.append(0)
            for m in model_preds:
                model_preds[m].append(0)
        if (i + 1) % 100 == 0:
            print("  %d/500 (errors:%d)" % (i + 1, errors))

    print("")
    print("=== FINAL RESULTS ===")
    prev = {"ENSEMBLE": 0.656, "BERT": 0.556, "TFIDF": 0.646, "NAIVE_BAYES": 0.634}
    for name, preds in [("ENSEMBLE", ens_preds)] + [
        (m.upper(), p) for m, p in model_preds.items()
    ]:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        diff = acc - prev.get(name, 0.0)
        arrow = "UP  +%.3f" % diff if diff > 0 else "DOWN %.3f" % diff
        print(
            "%-14s  acc=%.4f  f1=%.4f  prec=%.4f  [%s]"
            % (name, acc, f1, prec, arrow)
        )
    print("Errors: %d/500" % errors)
    ens_acc = accuracy_score(labels, ens_preds)
    print("")
    print("Previous best: 0.656")
    print("This run:      %.4f" % ens_acc)
    print("Status: %s" % ("IMPROVED" if ens_acc > 0.656 else "BELOW BEST"))


if __name__ == "__main__":
    main()
