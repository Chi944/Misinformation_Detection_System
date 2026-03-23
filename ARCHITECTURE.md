# System Architecture

## Overview

The misinformation detection system uses a three-model ensemble with
fuzzy logic calibration, an optional LLM judge, and an online feedback loop.

```
Input Text
    |
    +---> BERT Classifier (10%)      --+
    |     bert-base-uncased             |
    |     PyTorch                       |
    |                                   |
    +---> TF-IDF DNN (80%)           --+--> Weighted Ensemble
    |     word + char n-grams           |    --> Fuzzy Logic
    |     TensorFlow/Keras              |    --> LLM Judge (optional)
    |                                   |    --> Final Verdict
    +---> Naive Bayes (10%)          --+
          ComplementNB
          scikit-learn
```

---

## Components

### BERT Classifier (src/models/bert_classifier.py)
- Base model: bert-base-uncased (110M parameters)
- Fine-tuned on: ISOT real news + LIAR fact-checks + synthetic patterns
- Architecture: BertModel -> Dropout(0.3) -> Linear(768, 2)
- Trained on: Google Colab T4 GPU
- Val accuracy: 0.620

### TF-IDF DNN (src/models/tfidf_model.py)
- Features: 50k word n-grams + 30k char n-grams = 80k total
- Architecture: Dense(256, relu) -> Dropout(0.4) -> Dense(128, relu) -> Dense(2, softmax)
- Vectorizer: saved as models/tfidf_vectorizer.joblib
- Val accuracy: 0.656

### Naive Bayes (src/models/naive_bayes_model.py)
- Type: ComplementNB with online learning (partial_fit)
- Supports incremental updates via feedback loop
- Val accuracy: 0.634

### Ensemble (src/models/ensemble_detector.py)
- Weights: BERT=0.1, TF-IDF=0.8, NB=0.1
- Weights optimised via grid search on 10,000 val samples
- Ensemble accuracy: 0.682

### Fuzzy Logic (src/fuzzy/)
- Engine: Manual Mamdani inference (Python 3.12 compatible)
- Rules: 18 rules combining model confidences
- Inputs: bert_confidence, tfidf_confidence, nb_confidence
- Output: fuzzy_score (0.0 to 1.0)

### LLM Judge (src/evaluation/llm_judge.py)
- Backend: Local Ollama (no external API)
- Model: mistral (fallback: llama3, llama2)
- Optional: system works without it

### Feedback Loop (src/feedback/)
- Algorithm: Backpropagation with EWC regularisation
- EWC lambda: 0.1
- Prevents catastrophic forgetting on new data
- Storage: feedback.db (SQLite)

### Domain Credibility (src/utils/domain_credibility.py)
- Database: data/domain_reputation.json (80+ domains)
- Adjusts ensemble probability by up to 10% based on source
- Reuters/Nature: highly credible -> nudges toward credible
- Infowars/NaturalNews: unreliable -> nudges toward misinfo

### Explainability (src/utils/explainability.py)
- TF-IDF: top words by TF-IDF score
- NB: top words by log probability contribution
- Output added to prediction when explain=True

---

## Data Flow

```
predict(text, url=None, explain=False)
    |
    +--> BERT.predict(text)        -> bert_prob
    +--> TFIDFModel.predict(text)  -> tfidf_prob
    +--> NaiveBayes.predict(text)  -> nb_prob
    |
    +--> Ensemble.combine(bert_prob, tfidf_prob, nb_prob, weights)
    |        -> ensemble_prob
    |
    +--> FuzzyEngine.evaluate(bert_prob, tfidf_prob, nb_prob)
    |        -> fuzzy_score
    |
    +--> [optional] LLMJudge.evaluate(text)
    |        -> llm_verdict
    |
    +--> [if url] DomainCredibility.adjust_probability(ensemble_prob, url)
    |        -> adjusted_prob
    |
    +--> [if explain] Explainability.explain(text, tfidf, nb)
    |        -> top_words
    |
    +--> Result dict:
            ensemble_probability  float 0-1
            verdict               CREDIBLE / SUSPICIOUS / MISINFORMATION
            model_breakdown       per-model confidences
            fuzzy_score           float 0-1
            source_credibility    domain score and label (if url given)
            explanation           top words (if explain=True)
            llm_judge             LLM verdict (if Ollama running)
```

---

## Model Files

| File | Size | Description |
|---|---|---|
| models/bert_classifier.pt | 438 MB | BERT fine-tuned weights |
| models/tfidf_model.keras | 246 MB | TF-IDF DNN weights |
| models/tfidf_vectorizer.joblib | 5 MB | Fitted TF-IDF vectorizer |
| models/naive_bayes.pkl | 8 MB | Naive Bayes model |
| models/nb_vectorizer.pkl | 1 MB | NB vectorizer |

---

## Python 3.12 Compatibility

scikit-fuzzy uses the removed imp module. This is patched via
src/utils/skfuzzy_compat.py which must be imported before skfuzzy.
The FuzzyEngine handles this automatically.
