"""Collect per-model predictions on val set and grid search optimal ensemble weights."""
import os
import sys
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Cap val samples for speed (full val can be 12k+)
MAX_VAL = 500


def main():
    from src.detector import MisinformationDetector

    detector = MisinformationDetector(config="config.yaml", fast_mode=False)

    with open("data/val.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) > MAX_VAL:
        rows = rows[:MAX_VAL]
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    print("Collecting predictions on %d val samples..." % len(texts))
    bert_probs = []
    tfidf_probs = []
    nb_probs = []
    errors = 0

    for i, t in enumerate(texts):
        try:
            r = detector.predict(t)
            bd = r["model_breakdown"]
            bert_probs.append(bd["bert"]["confidence"])
            tfidf_probs.append(bd["tfidf"]["confidence"])
            nb_probs.append(bd["naive_bayes"]["confidence"])
        except Exception:
            errors += 1
            bert_probs.append(0.5)
            tfidf_probs.append(0.5)
            nb_probs.append(0.5)
        if (i + 1) % 500 == 0:
            print("  %d/%d" % (i + 1, len(texts)))

    print("Errors: %d" % errors)

    bert_probs = np.array(bert_probs)
    tfidf_probs = np.array(tfidf_probs)
    nb_probs = np.array(nb_probs)
    labels = np.array(labels)

    base = tempfile.gettempdir()
    np.save(os.path.join(base, "bert_probs.npy"), bert_probs)
    np.save(os.path.join(base, "tfidf_probs.npy"), tfidf_probs)
    np.save(os.path.join(base, "nb_probs.npy"), nb_probs)
    np.save(os.path.join(base, "val_labels.npy"), labels)
    print("Predictions saved to %s" % base)

    for name, probs in [
        ("BERT", bert_probs),
        ("TFIDF", tfidf_probs),
        ("NB", nb_probs),
    ]:
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        print("%s individual acc: %.4f" % (name, acc))

    # Grid search
    print("")
    print("Grid searching optimal weights (step 0.1)...")
    best_acc = 0
    best_f1 = 0
    best_weights = (0.5, 0.3, 0.2)
    results = []

    for wb in range(0, 11):
        for wt in range(0, 11 - wb):
            wn = 10 - wb - wt
            if wn < 0:
                continue
            w_bert = wb / 10.0
            w_tfidf = wt / 10.0
            w_nb = wn / 10.0
            if abs(w_bert + w_tfidf + w_nb - 1.0) > 0.01:
                continue

            ensemble = (
                w_bert * bert_probs
                + w_tfidf * tfidf_probs
                + w_nb * nb_probs
            )
            preds = (ensemble >= 0.5).astype(int)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            results.append((acc, f1, w_bert, w_tfidf, w_nb))

            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_weights = (w_bert, w_tfidf, w_nb)

    results.sort(reverse=True)
    print("")
    print("Top 5 weight combinations:")
    print("%-8s %-8s %-8s %-10s %-10s" % ("BERT", "TFIDF", "NB", "Acc", "F1"))
    for acc, f1, wb, wt, wn in results[:5]:
        print("%-8.1f %-8.1f %-8.1f %-10.4f %-10.4f" % (wb, wt, wn, acc, f1))

    print("")
    print("Best weights: bert=%.1f  tfidf=%.1f  nb=%.1f" % best_weights)
    print("Best val acc: %.4f  f1: %.4f" % (best_acc, best_f1))

    current = 0.5 * bert_probs + 0.3 * tfidf_probs + 0.2 * nb_probs
    curr_acc = accuracy_score(labels, (current >= 0.5).astype(int))
    print("Current (0.5/0.3/0.2) val acc: %.4f" % curr_acc)
    print("Improvement: +%.4f" % (best_acc - curr_acc))

    # Apply to config and ensemble_detector
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["models"]["bert"]["weight"] = best_weights[0]
    cfg["models"]["tfidf"]["weight"] = best_weights[1]
    cfg["models"]["naive_bayes"]["weight"] = best_weights[2]
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("")
    print("config.yaml updated with best weights.")

    # Update ensemble_detector.py _get_active_weights base dict
    ed_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "models",
        "ensemble_detector.py",
    )
    import re
    with open(ed_path) as f:
        content = f.read()
    new_base = """        base = {
            "bert": %.2f,
            "tfidf": %.2f,
            "naive_bayes": %.2f,
        }""" % (
        best_weights[0],
        best_weights[1],
        best_weights[2],
    )
    content = re.sub(
        r'        base = \{\s*"bert": [\d.]+,\s*"tfidf": [\d.]+,\s*"naive_bayes": [\d.]+,\s*\}',
        new_base.strip(),
        content,
        count=1,
    )
    content = re.sub(r"bert: float = [\d.]+", "bert: float = %.2f" % best_weights[0], content, count=1)
    content = re.sub(r"tfidf: float = [\d.]+", "tfidf: float = %.2f" % best_weights[1], content, count=1)
    content = re.sub(r"naive_bayes: float = [\d.]+", "naive_bayes: float = %.2f" % best_weights[2], content, count=1)
    with open(ed_path, "w") as f:
        f.write(content)
    print("ensemble_detector.py updated (base weights + EnsembleWeights defaults).")
    print("Done.")


if __name__ == "__main__":
    main()
