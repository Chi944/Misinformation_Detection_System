"""Grid search ensemble weights with all 3 models (min 0.1 each). Saves best_weights.npy."""
import os
import sys
import csv
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.detector import MisinformationDetector
from sklearn.metrics import accuracy_score, f1_score


def main():
    print("Collecting val predictions with ALL 3 models working...")
    detector = MisinformationDetector(config="config.yaml", fast_mode=False)

    with open("data/val.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Cap for speed (full val ~12.5k); set to None to use all
    MAX_VAL = 50
    if MAX_VAL and len(rows) > MAX_VAL:
        import random
        random.seed(42)
        rows = random.sample(rows, MAX_VAL)
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    print("Using %d val samples" % len(texts))

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
        if (i + 1) % 1000 == 0:
            print("  %d/%d" % (i + 1, len(texts)))

    print("Errors: %d/%d" % (errors, len(texts)))

    for name, probs in [
        ("BERT", bert_probs),
        ("TFIDF", tfidf_probs),
        ("NB", nb_probs),
    ]:
        ps = np.array(probs)
        flat = np.sum(np.abs(ps - 0.5) < 0.01) / len(ps)
        preds = (ps >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        print("%s: acc=%.4f  flat(near 0.5)=%.1f%%" % (name, acc, flat * 100))

    bp = np.array(bert_probs)
    tp = np.array(tfidf_probs)
    np_ = np.array(nb_probs)
    lab = np.array(labels)

    print("")
    print("Grid search (all models must have >= 0.1 weight)...")
    best_acc = 0
    best_w = (0.5, 0.3, 0.2)
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
    print("Top 5 combinations (all models >= 0.1 weight):")
    for acc, f1, wb, wt, wn in results[:5]:
        print("  bert=%.1f tfidf=%.1f nb=%.1f  acc=%.4f f1=%.4f" % (wb, wt, wn, acc, f1))
    print("")
    print(
        "Best: bert=%.1f tfidf=%.1f nb=%.1f  val_acc=%.4f"
        % (best_w[0], best_w[1], best_w[2], best_acc)
    )

    out_dir = tempfile.gettempdir()
    path = os.path.join(out_dir, "best_weights.npy")
    np.save(path, np.array(best_w))
    print("Saved best weights to %s" % path)
    # Also save to project for Step 4
    project_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "best_weights.npy",
    )
    np.save(project_path, np.array(best_w))
    print("Also saved to %s" % project_path)


if __name__ == "__main__":
    main()
