<#
.SYNOPSIS
    One-shot setup of FleshForge on Windows. No admin rights needed.

.DESCRIPTION
    Everything goes inside the repo folder; nothing is installed system-wide:

      .venv\          Python 3.12 virtualenv with FleshForge and its dependencies
      tools\colmap\   the official COLMAP CUDA release for Windows

    Then it runs fleshforge-check. Safe to re-run: finished steps are skipped.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#>
param(
    [string]$ColmapVersion = "4.2.1",
    # Download COLMAP again even if tools\colmap exists.
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# Native commands run with "Continue": under "Stop", Windows PowerShell 5.1 turns anything a
# program writes to stderr (a pip warning) into a terminating error. $LASTEXITCODE decides.
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$Exe $($Arguments -join ' ')' failed (exit $LASTEXITCODE)" }
}

Step "Python 3.12 virtualenv"
if (-not (Test-Path $VenvPy)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        Fail "the 'py' launcher was not found. Install Python 3.12: winget install -e --id Python.Python.3.12 --scope user"
    }
    # 3.10 to 3.12 work; open3d has no wheels for 3.13+
    Invoke-Checked "py" @("-3.12", "-m", "venv", (Join-Path $Root ".venv"))
}
Invoke-Checked $VenvPy @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
Invoke-Checked $VenvPy @("-m", "pip", "install", "--quiet", "-e", "$Root[dev]")

Step "COLMAP $ColmapVersion (CUDA build)"
$Colmap = Join-Path $Root "tools\colmap"
if ($Force -and (Test-Path $Colmap)) { Remove-Item -Recurse -Force $Colmap }
if (-not (Test-Path (Join-Path $Colmap "COLMAP.bat"))) {
    $url = "https://github.com/colmap/colmap/releases/download/$ColmapVersion/colmap-x64-windows-cuda.zip"
    $zip = Join-Path $env:TEMP "fleshforge-colmap.zip"
    Write-Host "    downloading $url (about 400 MB)"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
    New-Item -ItemType Directory -Force -Path $Colmap | Out-Null
    Expand-Archive -Path $zip -DestinationPath $Colmap -Force
    Remove-Item -Force $zip
    if (-not (Test-Path (Join-Path $Colmap "COLMAP.bat"))) { Fail "the COLMAP zip did not contain COLMAP.bat" }
} else {
    Write-Host "    already in tools\colmap"
}

Step "Check"
Invoke-Checked $VenvPy @("-m", "fleshforge.tools.check")
Write-Host "`nIf dense stereo crashes on your GPU, FleshForge falls back to Docker: docker pull colmap/colmap:latest"
