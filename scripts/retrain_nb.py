"""Retrain Naive Bayes on full training data and save (joblib)."""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.models.naive_bayes_model import TFNaiveBayesWrapper

def main():
    print("Loading training data...")
    with open("data/train.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    print("Training NB on %d samples..." % len(texts))
    print("  credible: %d  misinfo: %d" % (labels.count(0), labels.count(1)))

    nb = TFNaiveBayesWrapper()
    nb.fit(texts, labels)
    # _save() called inside fit() writes models/naive_bayes.pkl and models/nb_vectorizer.pkl
    print("Saved: models/naive_bayes.pkl, models/nb_vectorizer.pkl")

    # Validate on val set
    if os.path.exists("data/val.csv"):
        with open("data/val.csv", encoding="utf-8") as f:
            val = list(csv.DictReader(f))[:500]
        vt = [r["text"] for r in val]
        vl = [int(r["label"]) for r in val]
        proba = nb.predict_proba_np(vt)
        pred_labels = [1 if p[1] >= 0.5 else 0 for p in proba]
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(vl, pred_labels)
        f1 = f1_score(vl, pred_labels, zero_division=0)
        print("NB val: acc=%.4f  f1=%.4f" % (acc, f1))

if __name__ == "__main__":
    main()
