# How to share your app anywhere for free using ngrok

## Option 1: ngrok (Recommended - instant, no deployment needed)

ngrok creates a public URL that tunnels to your local app.
Anyone with the URL can access it while your machine is running.

### Setup (one time)

1. Go to https://ngrok.com and create a free account
2. Download ngrok for Windows from https://ngrok.com/download
3. Extract ngrok.exe to your project folder
4. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
5. Run: `ngrok config add-authtoken 3BM0xQ4hMX63j26xDjwAmg7FD0O_397qFmxUDFFNQAVQTY6fr`

### Run your app and share it

Terminal 1 - Start the API:

```bash
cd C:\Users\Desto\orchids-projects\orchids-misinformation-detection-app
.venv\Scripts\activate
python api.py
```

Terminal 2 - Start ngrok tunnel:

```bash
ngrok http 8000
```

ngrok will show you a URL like:

`https://abc123.ngrok-free.app -> http://localhost:8000`

Share that URL with anyone. They can access:

- `https://abc123.ngrok-free.app/app` (web interface)
- `https://abc123.ngrok-free.app/predict` (API endpoint)
- `https://abc123.ngrok-free.app/health` (health check)

## Option 2: HuggingFace Spaces (Permanent free hosting)

Requires:

- huggingface.co account (free)
- model files uploaded (700MB+ total)

### Deploy steps

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/new-space
3. Choose Gradio or Docker as SDK
4. Push your code and model files
5. HuggingFace runs it 24/7 for free

### Limitations

- BERT runs on CPU only (free) -> 5-10 seconds per prediction
- Must upload all model files
- Sleeps after inactivity (wakes on request)

For GPU (faster BERT): paid plan required.

## Option 3: Google Colab (Good for demos)

1. Upload project to Google Drive
2. Open Colab: https://colab.research.google.com
3. Mount Drive and run API + ngrok
4. Share the ngrok URL

Resets after session ends (8-12 hours).
