"""Retrain TF-IDF 80k Keras model using the already-fitted vectorizer (no refit)."""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf

# Load the already-fitted vectorizer
print("Loading vectorizer...")
vecs = joblib.load("models/tfidf_vectorizer.joblib")
word_vec = vecs["word"]
char_vec = vecs["char"]

# Verify vectorizer shape
test_w = word_vec.transform(["test text"]).toarray()
test_c = char_vec.transform(["test text"]).toarray()
input_dim = test_w.shape[1] + test_c.shape[1]
print("Vectorizer feature dim: %d (should be 80000)" % input_dim)
assert input_dim == 80000, "Expected 80000 got %d" % input_dim

# Load training data
print("Loading training data...")
with open("data/train.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.seed(42)
random.shuffle(rows)
rows = rows[:20000]
texts = [r["text"] for r in rows]
labels = [int(r["label"]) for r in rows]
print(
    "Using %d samples (%d credible / %d misinfo)"
    % (len(rows), labels.count(0), labels.count(1))
)

# Build feature matrix in batches
print("Building feature matrix...")
X_list = []
batch = 2000
for i in range(0, len(texts), batch):
    bt = texts[i : i + batch]
    w = word_vec.transform(bt).toarray()
    c = char_vec.transform(bt).toarray()
    X_list.append(np.hstack([w, c]).astype(np.float32))
    print("  Transformed %d/%d" % (min(i + batch, len(texts)), len(texts)))

X = np.vstack(X_list)
y = np.array(labels, dtype=np.int32)
print("Feature matrix shape:", X.shape)

# Build Keras model matching input_dim
print("Building Keras model (input_dim=%d)..." % input_dim)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Train
print("Training...")
history = model.fit(
    X,
    y,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/tfidf_model.keras")
size_mb = os.path.getsize("models/tfidf_model.keras") / 1e6
print("Saved: models/tfidf_model.keras (%.1f MB)" % size_mb)
print("Input shape:", model.input_shape)
