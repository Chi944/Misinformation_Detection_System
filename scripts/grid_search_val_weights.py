"""Grid search ensemble weights on full val.csv; writes best weights to config.yaml."""
import csv
import os
import random
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import accuracy_score, f1_score

from src.detector import MisinformationDetector


def main():
    cap = os.environ.get("GRID_VAL_MAX", "").strip()
    detector = MisinformationDetector(config="config.yaml", fast_mode=False)

    with open("data/val.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if cap.isdigit() and int(cap) > 0 and len(rows) > int(cap):
        random.seed(42)
        rows = random.sample(rows, int(cap))
        print("(GRID_VAL_MAX=%s) using %d val rows" % (cap, len(rows)))

    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    print("Val set: %d samples" % len(texts))

    bert_p, tfidf_p, nb_p = [], [], []
    errors = 0
    for i, t in enumerate(texts):
        try:
            r = detector.predict(t)
            bd = r["model_breakdown"]
            bert_p.append(bd["bert"]["confidence"])
            tfidf_p.append(bd["tfidf"]["confidence"])
            nb_p.append(bd["naive_bayes"]["confidence"])
        except Exception:
            errors += 1
            bert_p.append(0.5)
            tfidf_p.append(0.5)
            nb_p.append(0.5)
        if (i + 1) % 1000 == 0:
            print("  %d/%d" % (i + 1, len(texts)))

    print("Errors: %d" % errors)
    bp = np.array(bert_p)
    tp = np.array(tfidf_p)
    np_ = np.array(nb_p)
    lab = np.array(labels)

    for name, probs in [("BERT", bp), ("TFIDF", tp), ("NB", np_)]:
        acc = accuracy_score(lab, (probs >= 0.5).astype(int))
        print("%s val acc: %.4f" % (name, acc))

    best_acc = 0.0
    best_w = (0.3, 0.4, 0.3)
    results = []
    for wb in range(1, 9):
        for wt in range(1, 10 - wb):
            wn = 10 - wb - wt
            if wn < 1:
                continue
            wb_, wt_, wn_ = wb / 10.0, wt / 10.0, wn / 10.0
            ens = wb_ * bp + wt_ * tp + wn_ * np_
            preds = (ens >= 0.5).astype(int)
            acc = accuracy_score(lab, preds)
            f1 = f1_score(lab, preds, zero_division=0)
            results.append((acc, f1, wb_, wt_, wn_))
            if acc > best_acc:
                best_acc = acc
                best_w = (wb_, wt_, wn_)

    results.sort(reverse=True)
    print("")
    print("Top 5 weight combinations:")
    for acc, f1, wb, wt, wn in results[:5]:
        print(
            "  bert=%.1f tfidf=%.1f nb=%.1f  acc=%.4f f1=%.4f"
            % (wb, wt, wn, acc, f1)
        )
    print(
        "Best: bert=%.1f tfidf=%.1f nb=%.1f  acc=%.4f"
        % (best_w[0], best_w[1], best_w[2], best_acc)
    )

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["models"]["bert"]["weight"] = best_w[0]
    cfg["models"]["tfidf"]["weight"] = best_w[1]
    cfg["models"]["naive_bayes"]["weight"] = best_w[2]
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("config.yaml updated")


if __name__ == "__main__":
    main()
