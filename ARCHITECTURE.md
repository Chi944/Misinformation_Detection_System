# Architecture

## Pipeline
Input Text
|
+------------------+------------------+
|                  |                  |
BERTMisinformation   TFIDFModel     TFNaiveBayes
Classifier         (Keras DNN)      Wrapper
(bert-base-uncased)  (word+char      (Complement
PyTorch             TF-IDF)          NB)
|                  |                  |
+------------------+------------------+
|
EnsembleDetector
BERT 50% + TFIDF 30% + NB 20%
weights recalibrated each cycle
|
FuzzyMisinformationEngine
18 rules, manual Mamdani
Python 3.12 safe (no ControlSystem)
|
LLMJudge
Ollama llama3 local
no API key required
|
BackpropFeedbackLoop
EWC + online learning
hill-climbing thresholds

## Models

### BERT (src/models/bert_classifier.py)
- Base: bert-base-uncased
- Architecture: BertModel + Dropout(0.3) + Linear(768, 2)
- Framework: PyTorch
- Training: AdamW, lr=2e-5, warmup steps
- Output: 2-class logits (credible / misinformation)

### TF-IDF DNN (src/models/tfidf_model.py)
- Input: word TF-IDF (50k features) + char TF-IDF (30k) + extra features
- Architecture: Dense(256, relu) -> Dropout -> Dense(128, relu) -> Dense(2)
- Framework: TensorFlow/Keras
- Online update: Keras model.fit() with small batches

### Naive Bayes (src/models/naive_bayes_model.py)
- Model: ComplementNB (handles class imbalance)
- Vectorizer: TF-IDF (sklearn)
- Online learning: partial_fit() for feedback loop updates
- Wrapper: TFNaiveBayesWrapper (TF-compatible interface)

### Ensemble (src/models/ensemble_detector.py)
- Default weights: BERT=0.50, TF-IDF=0.30, NB=0.20
- Combination: weighted average of softmax probabilities
- Recalibration: weights updated each feedback cycle

## Fuzzy Logic Engine (src/fuzzy/)

Manual Mamdani inference — Python 3.12 compatible:
`python
# skfuzzy_compat.py patches the removed imp module
import src.utils.skfuzzy_compat  # must come before import skfuzzy
import skfuzzy as fuzz
`

- 6 antecedents: source_credibility, bert_confidence, tfidf_confidence,
  nb_confidence, model_agreement, feedback_score
- 1 consequent: misinfo_score
- 18 rules covering LOW/MEDIUM/HIGH input combinations
- MFs: trimf LOW=[0,0,0.45] MEDIUM=[0.35,0.5,0.65] HIGH=[0.55,1,1]
- Output MFs (trapmf):
  credible=[0,0,0.25,0.4]
  suspicious=[0.3,0.45,0.55,0.7]
  misinformation=[0.6,0.75,1,1]
- Defuzzification: centroid method
- Returns 0.5 on any exception (safe fallback)

## LLM Judge (src/evaluation/llm_judge.py)

Local Ollama integration — no API key:

- Endpoint: http://localhost:11434/api/generate
- Model: llama3 (configurable in config.yaml)
- Evaluates: independent verdict, per-model judgment, fuzzy calibration
- Fallback: returns FALLBACK_VERDICT dict if Ollama is not running
- Temperature: 0.1, max tokens: 400

## Feedback Loop (src/feedback/backprop_loop.py)

BackpropFeedbackLoop.run_cycle() — 9 steps:

1. Forward pass: predict() for each sample
2. Ground truth: LLM judge or supplied true_labels
3. Error signals: per-model error computation
4. Backward pass: online update for high-error samples only
5. Fuzzy threshold hill-climbing
6. Ensemble weight recalibration
7. Persist all samples to FeedbackStore (SQLite)
8. Trend check: write RETRAIN_REQUIRED.flag if F1 < 0.75 x 3 cycles
9. Git commit cycle results via GitManager

## EWC Regularisation

Prevents catastrophic forgetting during BERT online updates:
Loss = CrossEntropy(pred, label)
+ 0.1 * 0.5 * sum(fisher_i * (param_i - stored_param_i)^2)

Fisher information approximated as uniform (ones tensors).
Lambda = 0.1 (configured in config.yaml feedback.ewc_lambda).

## Calibration (src/training/calibration.py)

Post-hoc temperature scaling on validation logits:
- Learns single T parameter minimising NLL (gradient descent, 50 steps)
- Applied: logits / T -> sigmoid -> calibrated probability
- One TemperatureScaler per model in EnsembleCalibrator
