"""Retrain TF-IDF Keras model to match current vectorizer dimension (word+char)."""
import os
import sys
import csv
import random
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

def main():
    vecs = joblib.load("models/tfidf_vectorizer.joblib")
    w = vecs["word"].transform(["test"]).toarray()
    c = vecs["char"].transform(["test"]).toarray()
    dim = w.shape[1] + c.shape[1]
    print("Vectorizer dim:", dim)

    with open("data/train.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(42)
    random.shuffle(rows)
    rows = rows[:20000]
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    print("Building features for %d samples..." % len(texts))
    X_list = []
    for i in range(0, len(texts), 2000):
        bt = texts[i : i + 2000]
        wf = vecs["word"].transform(bt).toarray()
        cf = vecs["char"].transform(bt).toarray()
        X_list.append(np.hstack([wf, cf]).astype(np.float32))
    X = np.vstack(X_list)
    y = np.array(labels, dtype=np.int32)
    print("Feature matrix:", X.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    print("Training 2 epochs...")
    model.fit(X, y, epochs=2, batch_size=128, validation_split=0.1, verbose=1)

    model.save("models/tfidf_model.keras")
    size = os.path.getsize("models/tfidf_model.keras") / 1e6
    print("Saved: %.1f MB  input=%s" % (size, model.input_shape))
    assert model.input_shape[-1] == dim, "WRONG DIM after save!"
    print("TF-IDF model correctly saved with dim=%d" % dim)


if __name__ == "__main__":
    main()
