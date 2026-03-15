"""Diagnose TF-IDF model vs vectorizer dimension mismatch."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    import tensorflow as tf
    import joblib

    print("=== tfidf_model.keras ===")
    model = tf.keras.models.load_model("models/tfidf_model.keras")
    print("Input shape:", model.input_shape)
    print("File size: %.1f MB" % (os.path.getsize("models/tfidf_model.keras") / 1e6))

    print("")
    print("=== tfidf_vectorizer.joblib ===")
    vecs = joblib.load("models/tfidf_vectorizer.joblib")
    w = vecs["word"].transform(["test"]).toarray()
    c = vecs["char"].transform(["test"]).toarray()
    X = np.hstack([w, c])
    print("Vectorizer output shape:", X.shape)

    print("")
    match = model.input_shape[-1] == X.shape[1]
    print("Match:", match)
    if not match:
        print(
            "MISMATCH: model expects %d but vectorizer gives %d"
            % (model.input_shape[-1], X.shape[1])
        )

    print("")
    print("=== TFIDFModel predict ===")
    from src.models.tfidf_model import TFIDFModel
    m = TFIDFModel(config={})
    print("Keras input dim:", m.model.input_shape[-1] if m.model else "None")
    preds = m.predict(["Scientists confirm findings.", "SHOCKING cover-up exposed!"])
    print("Predictions:", [round(float(p), 4) for p in preds])
    print("Are all 0.5?", all(abs(float(p) - 0.5) < 0.001 for p in preds))


if __name__ == "__main__":
    main()
