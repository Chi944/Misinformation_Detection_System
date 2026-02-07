# Running in GitHub Codespaces

The app is set up to run in GitHub Codespaces with minimal setup.

## 1. Open in Codespaces

- Push this repo to GitHub.
- Open the repo on GitHub → click **Code** → **Codespaces** → **Create codespace on main**.

## 2. After the container is ready

The devcontainer installs Python deps and NLTK data. Then run:

```bash
# Optional: verify app compiles and runs (synthetic data + train + predict)
python scripts/verify.py

# Organise data: copy FakeNewsNet.csv, train.tsv, valid.tsv, test.tsv into data/raw
python scripts/organise_data.py

# Train using Hugging Face dataset + data in data/raw (run once)
./run.sh train

# Start the API (port 5000)
./run.sh api
```

In the **Ports** tab, forward **5000** and open the API URL.

## 3. Frontend (optional)

In a **new terminal** in the same codespace:

```bash
cd frontend
npm install
npm run dev
```

Forward port **5173** and open the frontend URL.

## Data layout

| Location | Purpose |
|----------|--------|
| `data/raw/` | Raw data: `FakeNewsNet.csv`, `train.tsv`, `valid.tsv`, `test.tsv` (copied from repo root by `organise_data.py`) |
| `data/processed/` | Preprocessed train/val/test CSVs written by `prepare_data()` |
| `models/` | Trained models (`.pkl`, `bert_model/`) |

Training loads:

1. **Hugging Face:** `datasets.load_dataset("kasperdinh/fake-news-detection")`
2. **data/raw:** Any of `FakeNewsNet.csv`, `train.tsv`, `valid.tsv`, `test.tsv`

If Hugging Face or local files are missing, the app falls back to synthetic data.
