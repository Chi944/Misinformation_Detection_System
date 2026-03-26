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

### Deployment Target

- Production: HuggingFace Spaces (Docker)
  https://huggingface.co/spaces/werty3684/misinformation-detector
- Ports:
  - Local run: `8000`
  - Production container: `7860`

### Deploy to GitHub + HuggingFace

From project root:

```powershell
.\deploy.ps1 "Your commit message here"
```

This script:
- commits/pushes current repo to `origin main`
- syncs deployment artifacts to `C:\Users\Desto\hf-deploy`
- commits/pushes `hf-deploy` to `huggingface main`

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

### Option C: API + Frontend (local)

```bash
python api.py
```

Open:
- API: http://localhost:8000/health
- Frontend: http://localhost:8000/app

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

## Running Locally with Ollama

1. Open Terminal 1 and run: `ollama serve`
   - Keep this terminal open at all times
2. First time only - Open Terminal 2 and run: `ollama pull mistral`
3. Open Terminal 3 and run:

```bash
cd C:\Users\Desto\Desktop\Murdoch\ICT206\misinformation-detection-app
.venv\Scripts\activate
python api.py
```

4. Open browser at http://localhost:8000/app

Notes:
- Ollama must be running BEFORE starting `api.py` otherwise the LLM judge will be disabled.
- The app still works without Ollama but LLM reasoning will be unavailable.
- `ollama pull mistral` only needs to be run once.

---

## Verify Everything Works

```bash
python scripts/smoke_test.py --synthetic
python -m pytest tests/ --timeout=60
```

Expected:
- `Smoke test results: 8/8 passed`
- `54 passed, 5 skipped, 0 failed`
