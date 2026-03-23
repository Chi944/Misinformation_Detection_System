# Running the Misinformation Detection System

This guide covers everything needed to run the system using the
pre-trained ensemble model. No training required.

---

## Prerequisites

### 1. Python 3.12

Check your Python version:

```bash
python --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/Chi944/Misinformation_Detection_System.git
cd Misinformation_Detection_System
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Model Files Exist

```bash
python -c "import os; files=['models/bert_classifier.pt','models/tfidf_model.keras','models/tfidf_vectorizer.joblib','models/naive_bayes.pkl','models/nb_vectorizer.pkl']; \
[(print(('FOUND %.0f MB' % (os.path.getsize(f)/1e6) if os.path.exists(f) else 'MISSING') + '  ' + f)) for f in files]"
```

---

## Running the System

### Option A: Python Shell

```python
from src.detector import MisinformationDetector
detector = MisinformationDetector(config="config.yaml")
result = detector.predict("Scientists confirm vaccine safety in peer reviewed study.")
print(result["crisp_label"], result["ensemble_probability"])
```

### Option B: Quick CLI

```bash
python -c "from src.detector import MisinformationDetector; d=MisinformationDetector(config='config.yaml'); print(d.predict('SHOCKING cover-up exposed by insiders today', explain=True))"
```

---

## Advanced Usage

### Predict with Source URL

```python
result = detector.predict(
    "Scientists confirm findings",
    url="https://reuters.com/health/article",
)
print(result["source_credibility"])
```

### Predict with Explainability

```python
result = detector.predict(
    "SHOCKING cover-up exposed by whistleblower",
    explain=True,
)
print(result["explanation"]["summary"])
```

---

## Optional: Enable LLM Judge

```bash
ollama serve
ollama pull mistral
curl http://localhost:11434/api/tags
```

---

## Verify Everything Works

```bash
python scripts/smoke_test.py --synthetic
python -m pytest tests/ --timeout=60
```

Expected:
- `Smoke test results: 8/8 passed`
- `54 passed, 5 skipped, 0 failed`
