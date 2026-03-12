# Misinformation Detection App

A credibility audit system that classifies text and URLs as **Credible** or **Misinformation**, with metrics for sensationalism, political bias, source credibility, factuality index, and flagged terms.

---

## How to run

- **GitHub Codespaces:** See **[CODESPACES.md](CODESPACES.md)**. The app is set up to run there with the Hugging Face dataset and data in `data/raw/`.
- **Local (Windows):** See **[RUN.md](RUN.md)** for a step-by-step guide.

Short version:

1. **Install Python** from [python.org](https://www.python.org/downloads/) (check “Add to PATH”).
2. **In PowerShell:** `cd C:\path\to\misinformation-detection-app`
3. **Setup once:** `.\setup.ps1` (installs only `requirements.txt`).
4. **Train:** `.\.venv\Scripts\python main.py --train` (for best accuracy use `--train --tune`)
5. **Start API:** `.\.venv\Scripts\python main.py --api --port 5000`
6. **Optional frontend:** In a new terminal, `cd frontend` then `npm install` and `npm run dev`; open http://localhost:5173.
7. **LLM Judge (optional but recommended):** Install and start Ollama locally, then run:
   `ollama pull llama3`

---

## Quick Start (manual venv)

If you prefer to set up the venv yourself:

**Option A – Create venv and activate (PowerShell)**

```powershell
# Create virtual environment
python -m venv .venv

# Activate (try Scripts first; if that fails, use bin)
.\.venv\Scripts\Activate.ps1
# If "cannot be loaded" or "not recognised", try:
# .\.venv\bin\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the API
python main.py --api --port 5000
```

**Option B – Use venv without activating (any shell)**

If activation fails (e.g. no `Scripts` or `bin`), use the venv’s Python and pip directly. **Run from the project folder** (where `requirements.txt` and `main.py` are):

```powershell
cd C:\path\to\misinformation-detection-app

# Create venv (only once)
python -m venv .venv

# Install using venv's pip (Scripts = Windows, bin = MSYS2/WSL-style)
.\.venv\Scripts\pip install -r requirements.txt
# Or: .\.venv\bin\pip install -r requirements.txt

# Run the API with venv's Python
.\.venv\Scripts\python main.py --api --port 5000
# Or: .\.venv\bin\python main.py --api --port 5000
```

You should see `Starting API server at http://0.0.0.0:5000`. Keep this terminal open.

> **Note:** Train models first (`python main.py --train`) so predictions work. For higher accuracy, train with hyperparameter tuning: `python main.py --train --tune`. Without models, the API starts but `/predict` will return an error.

**Troubleshooting:** If you see *"externally-managed-environment"*, your system Python blocks global `pip install`. Always use a venv (Option A or B above). If `.\.venv\Scripts\Activate.ps1` is "not recognised", your venv may use `bin` instead of `Scripts` (e.g. MSYS2/Git Bash Python)—use `.\.venv\bin\Activate.ps1` or Option B.

**SSL / certificate errors (MSYS2 Python):** If you get `CERTIFICATE_VERIFY_FAILED` or `Failed building wheel for cmake`:

1. **Bypass SSL for pip** (project folder, use your venv’s pip):
   ```powershell
   .\.venv\bin\pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```
2. **If cmake still fails:** Install CMake in MSYS2 (`pacman -S mingw-w64-x86_64-cmake`), then retry the pip command above.
3. **Alternative:** Use [Python from python.org](https://www.python.org/downloads/) (not MSYS2), create a new venv with that Python, and run the same pip/run commands—SSL and wheels usually work without extra steps.
4. **Fallback (no Hugging Face data):** Install minimal deps so the app runs with FakeNewsNet only:  
   `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-minimal.txt`  
   (Use your venv’s pip and the same trusted-host flags if you get SSL errors.)

**“Cannot compile Python.h” / python-devel (MSYS2 Python):** Your Python is building packages from source and is missing development headers. Either:

- **Recommended:** Install [Python from python.org](https://www.python.org/downloads/), add it to PATH, remove or rename the old `.venv`, then run `.\setup.ps1` again. No compilation needed.
- **Or stay on MSYS2:** In an **MSYS2 MinGW 64-bit** terminal run:  
  `pacman -S mingw-w64-x86_64-python-devel mingw-w64-x86_64-gcc`  
  Then in the project folder:  
  `.\.venv\bin\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-minimal.txt`

### 2. Frontend (React + Vite)

Open a **second** terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser. The frontend proxies API requests to the backend on port 5000.

---

## How to Run the App

| Command | Description |
|--------|-------------|
| `python main.py --api` | Start API server (default port 5000) |
| `python main.py --api --port 8080` | Start API on a custom port |
| `python main.py --demo` | Run CLI inference demo |
| `python main.py --train` | Train the TF-IDF + Logistic Regression model |
| `python main.py --evaluate` | Evaluate the trained model on test set |
| `python main.py --all` | Full pipeline: train → evaluate → demo |

**Data sources (automatic)**

- **Hugging Face:** `kasperdinh/fake-news-detection` (loaded via `datasets`)
- **FakeNewsNet.csv:** In project root or `data/raw/` (columns: title, real)
- Falls back to synthetic data if neither is available

---

## How to Train the Model Further

1. **Prepare data**  
   Place CSV files in `data/raw/`. The preprocessing expects columns such as `text` and `label` (0 = Credible, 1 = Misinformation). Adjust `src/data_preprocessing.py` if your schema differs.

2. **Train**
   ```powershell
   python main.py --train
   ```
   The model is saved to `models/tfidf_logistic.pkl`.

3. **Optional: Hyperparameter tuning**
   ```powershell
   python main.py --train --tune
   ```

4. **Re-evaluate after training**
   ```powershell
   python main.py --evaluate
   ```
   This runs the trained model on the test set and writes visualizations to `results/`.

---

## Project Structure

```
misinformation-detection-app/
├── main.py              # CLI entry: train, evaluate, demo, api
├── requirements.txt     # Python dependencies
├── src/
│   ├── config.py        # Paths, labels, constants
│   ├── data_preprocessing.py
│   ├── traditional_ml.py # TF-IDF + Logistic Regression (single model)
│   ├── evaluation.py    # Metrics, plots
│   ├── explainability.py # LIME
│   ├── credibility_audit.py # Sensationalism, bias, source credibility, etc.
│   └── inference.py     # Flask API, URL fetch
├── frontend/            # React + Vite UI
│   └── src/
│       ├── App.tsx
│       └── index.css
├── data/raw/            # Training data
├── models/              # Saved model (tfidf_logistic.pkl)
└── results/             # Evaluation outputs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, models loaded |
| GET | `/models` | Available models and config |
| POST | `/predict` | Single prediction (text or url) |
| POST | `/predict/batch` | Batch predictions |
| POST | `/explain` | Prediction with LIME explanation |

**Example `/predict` request:**

```json
{ "text": "BREAKING: Shocking truth revealed!" }
```

or

```json
{ "url": "https://example.com/article", "header": "Optional headline" }
```

---

## Credibility Audit Metrics

- **Sensationalism** – Caps, exclamation marks, sensational terms
- **Political Bias** – Left/right/neutral from keyword indicators
- **Source Credibility** – Domain-based score when URL is provided
- **Factuality Index** – `1 - P(misinformation)`
- **Flagged Terms** – Words that reduce credibility (from LIME + sensational list)
