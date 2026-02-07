# Step-by-step: How to run the app (minimal setup)

Follow these steps in order. Only one small set of Python packages is installed.

---

## Step 1: Install Python (if needed)

1. Go to **https://www.python.org/downloads/**
2. Download the **Windows installer** (e.g. Python 3.12).
3. Run it. **Important:** check **“Add python.exe to PATH”**, then click “Install Now”.
4. Close and reopen PowerShell so `py` or `python` is available.

*(Using Python from python.org avoids “Python.h”, SSL, and cmake errors. Do not use MSYS2/MinGW Python for this project.)*

---

## Step 2: Open PowerShell and go to the project folder

```powershell
cd C:\Users\Desto\orchids-projects\orchids-misinformation-detection-app
```

You must be in the folder that contains `main.py` and `requirements-minimal.txt`.

---

## Step 3: Create a virtual environment and install dependencies (minimal only)

Run the setup script once:

```powershell
.\setup.ps1
```

**If you get “script cannot be loaded”:** run this first, then try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
```

**What this does:** Creates a `.venv` folder and installs only the packages in `requirements-minimal.txt` (no Hugging Face, no extra tools). You should see “Done. Next steps:” at the end.

**If setup.ps1 fails:** do it manually:

```powershell
py -3.12 -m venv .venv
# Use Scripts if it exists, otherwise bin (e.g. if venv was created by MSYS2):
.\.venv\Scripts\pip install -r requirements.txt
# If the above fails with "not recognized", use:
.\.venv\bin\pip install -r requirements.txt
```

---

## Step 4: Train the model (first time only)

**If your venv has a `Scripts` folder** (python.org Python):

```powershell
.\.venv\Scripts\python main.py --train
```

**If you get “not recognized”**, your venv uses `bin` instead. Use:

```powershell
.\.venv\bin\python main.py --train
```

- **If you have `FakeNewsNet.csv`** in the project folder: training uses that file.
- **If you don’t:** the app uses synthetic data and you’ll see “Creating synthetic dataset…”.
- Wait until you see **“TRAINING COMPLETE”**. Models are saved in the `models` folder.

---

## Step 5: Start the backend API

In the **same** PowerShell (still in the project folder). Use **Scripts** or **bin** to match your venv (see Step 4):

```powershell
.\.venv\Scripts\python main.py --api --port 5000
```

If that fails with “not recognized”:

```powershell
.\.venv\bin\python main.py --api --port 5000
```

You should see:

- `Starting API server at http://0.0.0.0:5000`
- List of endpoints (e.g. GET /health, POST /predict)

Leave this window open. The API is now running.

---

## Step 6: Open the frontend (optional)

Open a **new** PowerShell window:

```powershell
cd C:\Users\Desto\orchids-projects\orchids-misinformation-detection-app\frontend
npm install
npm run dev
```

Then open in your browser: **http://localhost:5173**

*(If you don’t need the UI, you can skip this and call the API with curl or Postman.)*

---

## Step 7: Use the app

- **In the browser:** Go to http://localhost:5173, enter text or a URL, click “Run prediction”.
- **API only:**  
  `POST http://127.0.0.1:5000/predict` with body: `{"text": "Your headline or text here"}`

---

## Quick reference (after setup is done)

Run from the project folder. If `.\.venv\Scripts\...` gives “not recognized”, use `.\.venv\bin\...` instead.

| What you want      | Command |
|--------------------|--------|
| Train model        | `.\.venv\Scripts\python main.py --train` or `.\.venv\bin\python main.py --train` |
| Start API          | `.\.venv\Scripts\python main.py --api --port 5000` or `.\.venv\bin\python main.py --api --port 5000` |
| Start frontend     | `cd frontend` then `npm run dev` |

---

## If something goes wrong

- **“No such file or directory”**  
  Make sure you’re in the project folder:  
  `cd C:\Users\Desto\orchids-projects\orchids-misinformation-detection-app`

- **“externally-managed-environment”**  
  Always use the venv:  
  `.\.venv\Scripts\pip` and `.\.venv\Scripts\python` (not global `pip`/`python`).

- **“Python.h” / “python-devel” / SSL / cmake errors**  
  Use **Python from python.org** (Step 1), delete the old `.venv` folder, then run `.\setup.ps1` again.

- **API starts but “Model not found” on predict**  
  Run training first:  
  `.\.venv\Scripts\python main.py --train`
