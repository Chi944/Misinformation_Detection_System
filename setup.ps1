# Minimal setup: one requirements file only (no optional downloads).
# Use Python from https://www.python.org/downloads/ (not MSYS2) for reliable installs.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan

# Prefer Windows Python launcher (python.org)
$pythonExe = $null
foreach ($ver in @("3.12")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $pyOut = & py -$ver -c "import sys; print(sys.executable)" 2>$null
            if ($pyOut) { $pythonExe = $pyOut.Trim(); break }
        } catch {}
    }
}
if (-not $pythonExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $pythonExe = (Get-Command python).Source
}

if (-not $pythonExe) {
    Write-Host "ERROR: Install Python from https://www.python.org/downloads/ and check 'Add to PATH'" -ForegroundColor Red
    exit 1
}

if ($pythonExe -match "msys64|mingw64|MinGW") {
    Write-Host "WARNING: MSYS2 Python often fails to install packages. Use python.org Python instead." -ForegroundColor Yellow
}

# Create venv
$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating .venv ..." -ForegroundColor Green
    & $pythonExe -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$pipExe = Join-Path $venvPath "Scripts\pip.exe"
if (-not (Test-Path $pipExe)) { $pipExe = Join-Path $venvPath "bin\pip.exe" }
$pythonVenv = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $pythonVenv)) { $pythonVenv = Join-Path $venvPath "bin\python.exe" }

if (-not (Test-Path $pipExe)) {
    Write-Host "ERROR: pip not found in .venv" -ForegroundColor Red
    exit 1
}

# Single install: minimal only (no Hugging Face, no extra packages)
$trusted = "--trusted-host pypi.org", "--trusted-host pypi.python.org", "--trusted-host files.pythonhosted.org"
$req = Join-Path $ProjectRoot "requirements.txt"

Write-Host "Installing minimal dependencies (this is the only download) ..." -ForegroundColor Green
& $pipExe install --upgrade pip @trusted -r $req
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install failed. Use Python from python.org and run again." -ForegroundColor Red
    exit $LASTEXITCODE
}

$pyCmd = ".\.venv\Scripts\python"
if (-not (Test-Path (Join-Path $venvPath "Scripts\python.exe"))) {
    $pyCmd = ".\.venv\bin\python"
}
Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  1. $pyCmd main.py --train" -ForegroundColor White
Write-Host "  2. $pyCmd main.py --api --port 5000" -ForegroundColor White
Write-Host "  3. In another terminal: cd frontend; npm install; npm run dev" -ForegroundColor White
Write-Host ""
