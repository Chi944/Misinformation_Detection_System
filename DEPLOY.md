# Deployment (HuggingFace Spaces - Docker)

Production deployment platform:

- Space: https://huggingface.co/spaces/werty3684/misinformation-detector
- Runtime: Docker
- Production port: `7860`
- Local development port: `8000`

---

## Port behavior

The API is configured to run on:

- **Local:** `8000`
- **HuggingFace Spaces:** `7860` (or `$PORT` if provided by runtime)

`api.py` resolves port automatically:

- Uses `7860` when `SPACE_ID` is present
- Otherwise defaults to `8000`
- Honors `PORT` environment variable if set

---

## Local run

```bash
python api.py
```

Open:

- API health: http://localhost:8000/health
- Frontend: http://localhost:8000/app

---

## HuggingFace Spaces run (Docker)

The `Dockerfile` exposes port `7860` and starts:

```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

Use the Space URL:

- App: https://huggingface.co/spaces/werty3684/misinformation-detector

---

## Notes

- No tunneling tools are required.
- Ollama/LLM judge remains optional in production.

---

## Automated Deployment Script

Use the root script to deploy to both GitHub and HuggingFace in one run:

```powershell
.\deploy.ps1 "Your commit message here"
```

The script performs:

1. Commit + push main project to `origin/main`
2. Sync required files into `C:\Users\Desto\hf-deploy`:
   - `src/`
   - `models/`
   - `api.py`
   - `config.yaml`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
3. Commit + push `hf-deploy` to `huggingface/main`
