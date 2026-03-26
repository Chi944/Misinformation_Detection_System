param(
    [Parameter(Mandatory = $false, Position = 0)]
    [string]$CommitMessage = "Update"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# --- Paths and remotes ---
$MainProject = "C:\Users\Desto\Desktop\Murdoch\ICT206\misinformation-detection-app"
$HfDeploy = "C:\Users\Desto\hf-deploy"

$GitHubUrl = "https://github.com/Chi944/Misinformation_Detection_System"
$HfSpaceUrl = "https://huggingface.co/spaces/werty3684/misinformation-detector"

# --- Utility helpers ---
function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    try {
        & $Action
    } catch {
        Write-Host ""
        Write-Host "ERROR during: $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        throw
    }
}

function Invoke-Robocopy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )
    # /MIR keeps destination in sync, /NFL /NDL trims noise
    robocopy $Source $Destination /MIR /NFL /NDL /NJH /NJS /NP | Out-Host
    $code = $LASTEXITCODE
    # Robocopy success codes are 0..7
    if ($code -gt 7) {
        throw "robocopy failed from '$Source' to '$Destination' with exit code $code"
    }
}

function Invoke-GitCommitAndPush {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoPath,
        [Parameter(Mandatory = $true)]
        [string]$RemoteName,
        [Parameter(Mandatory = $true)]
        [string]$Branch
    )

    Push-Location $RepoPath
    try {
        git add -A
        if ($LASTEXITCODE -ne 0) { throw "git add failed in $RepoPath" }

        $status = git status --porcelain
        if ($LASTEXITCODE -ne 0) { throw "git status failed in $RepoPath" }

        if ([string]::IsNullOrWhiteSpace($status)) {
            Write-Host "No changes to commit in $RepoPath"
        } else {
            git commit -m $CommitMessage
            if ($LASTEXITCODE -ne 0) { throw "git commit failed in $RepoPath" }
        }

        git push $RemoteName $Branch
        if ($LASTEXITCODE -ne 0) { throw "git push failed in $RepoPath ($RemoteName/$Branch)" }
    } finally {
        Pop-Location
    }
}

function Copy-FileChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceFile,
        [Parameter(Mandatory = $true)]
        [string]$DestinationDir
    )
    if (-not (Test-Path $SourceFile)) {
        throw "Required file not found: $SourceFile"
    }
    if ([System.IO.Path]::GetFileName($SourceFile).ToLowerInvariant() -eq "readme.txt") {
        throw "README.txt is explicitly excluded from HF deploy sync: $SourceFile"
    }
    Copy-Item -Path $SourceFile -Destination $DestinationDir -Force
}

# --- Preflight ---
Invoke-Step -Name "Preflight checks" -Action {
    if (-not (Test-Path $MainProject)) { throw "Main project folder missing: $MainProject" }
    if (-not (Test-Path $HfDeploy)) { throw "HF deploy folder missing: $HfDeploy" }

    if (-not (Test-Path (Join-Path $MainProject ".git"))) {
        throw "Main project is not a git repo: $MainProject"
    }
    if (-not (Test-Path (Join-Path $HfDeploy ".git"))) {
        throw "HF deploy folder is not a git repo: $HfDeploy"
    }
}

# --- 1) Push main project to GitHub ---
Invoke-Step -Name "Push main project to GitHub (origin/main)" -Action {
    Invoke-GitCommitAndPush -RepoPath $MainProject -RemoteName "origin" -Branch "main"
}

# --- 2) Build frontend ---
Invoke-Step -Name "Build frontend (npm run build)" -Action {
    $frontendProjectDir = Join-Path $MainProject "frontend"
    if (-not (Test-Path $frontendProjectDir)) { throw "Missing frontend folder: $frontendProjectDir" }
    Push-Location $frontendProjectDir
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed in $frontendProjectDir" }
    } finally {
        Pop-Location
    }
}

# --- 3) Sync files to HF deploy workspace ---
Invoke-Step -Name "Sync project files to HF deploy folder" -Action {
    $srcDir = Join-Path $MainProject "src"
    $modelsDir = Join-Path $MainProject "models"
    $frontendDistDir = Join-Path $MainProject "frontend\dist"
    $dstSrc = Join-Path $HfDeploy "src"
    $dstModels = Join-Path $HfDeploy "models"
    $dstFrontendDist = Join-Path $HfDeploy "frontend\dist"

    if (-not (Test-Path $srcDir)) { throw "Missing source folder: $srcDir" }
    if (-not (Test-Path $modelsDir)) { throw "Missing models folder: $modelsDir" }
    if (-not (Test-Path $frontendDistDir)) { throw "Missing built frontend dist folder: $frontendDistDir" }

    # Ensure README.txt is never present in HF deploy workspace.
    $hfReadmeTxt = Join-Path $HfDeploy "README.txt"
    if (Test-Path $hfReadmeTxt) {
        Remove-Item -Path $hfReadmeTxt -Force
    }

    Invoke-Robocopy -Source $srcDir -Destination $dstSrc
    Invoke-Robocopy -Source $modelsDir -Destination $dstModels
    # Copy only built frontend assets to HuggingFace deploy workspace.
    # This guarantees node_modules is never copied.
    robocopy $frontendDistDir $dstFrontendDist /MIR /NFL /NDL /NJH /NJS /NP | Out-Host
    $frontendCopyCode = $LASTEXITCODE
    if ($frontendCopyCode -gt 7) {
        throw "robocopy failed from '$frontendDistDir' to '$dstFrontendDist' with exit code $frontendCopyCode"
    }

    Copy-FileChecked -SourceFile (Join-Path $MainProject "api.py") -DestinationDir $HfDeploy
    Copy-FileChecked -SourceFile (Join-Path $MainProject "config.yaml") -DestinationDir $HfDeploy
    Copy-FileChecked -SourceFile (Join-Path $MainProject "requirements.txt") -DestinationDir $HfDeploy
    Copy-FileChecked -SourceFile (Join-Path $MainProject "Dockerfile") -DestinationDir $HfDeploy

    # Fix Linux incompatible packages in hf-deploy requirements.txt
    $reqPath = "C:\Users\Desto\hf-deploy\requirements.txt"
    if (-not (Test-Path $reqPath)) {
        throw "Copied requirements.txt not found in HF deploy folder: $reqPath"
    }
    (Get-Content $reqPath) |
      Where-Object { $_ -notmatch "tensorflow-intel|datasets" } |
      ForEach-Object {
        $_ -replace 'requests==2.31.0','requests==2.32.2' `
           -replace 'tqdm==4.66.2','tqdm==4.66.3'
      } |
      Set-Content -Encoding UTF8 "$reqPath.tmp"
    Remove-Item $reqPath
    Rename-Item "$reqPath.tmp" $reqPath
    Write-Host "requirements.txt cleaned for Linux"
}

# --- 4) Push HF deploy workspace to HuggingFace remote ---
Invoke-Step -Name "Push HF deploy folder to HuggingFace (huggingface/main)" -Action {
    Invoke-GitCommitAndPush -RepoPath $HfDeploy -RemoteName "huggingface" -Branch "main"
}

# --- Done ---
Write-Host ""
Write-Host "Deployment complete." -ForegroundColor Green
Write-Host "GitHub:      $GitHubUrl"
Write-Host "HuggingFace: $HfSpaceUrl"
