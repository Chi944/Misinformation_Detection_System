"""Retrain TF-IDF Keras with 2 epochs and stronger dropout to reduce overfitting."""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import tensorflow as tf

def main():
    vecs = joblib.load("models/tfidf_vectorizer.joblib")
    word_vec = vecs["word"]
    char_vec = vecs["char"]

    with open("data/train.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.seed(42)
    random.shuffle(rows)
    rows = rows[:20000]
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    print("Building features...")
    X_list = []
    for i in range(0, len(texts), 2000):
        bt = texts[i : i + 2000]
        w = word_vec.transform(bt).toarray()
        c = char_vec.transform(bt).toarray()
        X_list.append(np.hstack([w, c]).astype(np.float32))
    X = np.vstack(X_list)
    y = np.array(labels, dtype=np.int32)
    print("Feature shape:", X.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(80000,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X,
        y,
        epochs=2,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    model.save("models/tfidf_model.keras")
    size_mb = os.path.getsize("models/tfidf_model.keras") / 1e6
    print("Saved: %.1f MB" % size_mb)
    for i, (loss, val_acc) in enumerate(zip(history.history["loss"], history.history["val_accuracy"]), 1):
        print("Epoch %d: loss=%.4f  val_accuracy=%.4f" % (i, loss, val_acc))

if __name__ == "__main__":
    main()
