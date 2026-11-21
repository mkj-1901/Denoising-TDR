<#
.SYNOPSIS
Run both the original Gaussian demo and the ambient‑mean demo on a small batch,
collect metrics, and store results.

.PARAMETER Batch
Sub‑directory under `data/` (e.g. clean, denoising_gaussian001, denoising_poisson_b1000b).

.EXAMPLE
.\run_batch.ps1 -Batch clean
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Batch
)

# ---- 0 – sanity -------------------------------------------------
$repoRoot = (Get-Location).Path
Write-Host "Repository root: $repoRoot"

$demoScript   = Join-Path $repoRoot "TDR_Demo\demo.py"
$ambientScript= Join-Path $repoRoot "TDR_Demo\ambient_demo.py"

if (!(Test-Path $demoScript)) {
    Write-Error "demo.py not found at $demoScript"
    exit 1
}
if (!(Test-Path $ambientScript)) {
    Write-Error "ambient_demo.py not found at $ambientScript"
    exit 1
}

$batchPath = Join-Path $repoRoot "data\$Batch"
if (!(Test-Path $batchPath)) {
    Write-Error "Batch folder does not exist: $batchPath"
    exit 1
}

# ---- 0. ensure results folder exists ----
$resultsDir = Join-Path $repoRoot "results"
if (!(Test-Path $resultsDir)) { New-Item -ItemType Directory -Path $resultsDir | Out-Null }

# ---- 1 – run original Gaussian demo --------------------------------
Write-Host "`n=== Running original Gaussian demo on batch [$Batch] ==="
python $demoScript $repoRoot > (Join-Path $resultsDir "${Batch}_demo.log") 2>&1

# ---- 2 – run ambient‑mean demo ------------------------------------
Write-Host "`n=== Running ambient‑mean demo on batch [$Batch] ==="
python $ambientScript $repoRoot > (Join-Path $resultsDir "${Batch}_ambient.log") 2>&1

# ---- 3 – copy generated PNGs ---------------------------------------
Copy-Item -Path (Join-Path $repoRoot "demo_result.png") -Destination (Join-Path $resultsDir "${Batch}_demo_result.png") -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $repoRoot "ambient_demo_result.png") -Destination (Join-Path $resultsDir "${Batch}_ambient_result.png") -Force -ErrorAction SilentlyContinue

Write-Host "`n=== Finished batch [$Batch] ==="
Write-Host "Logs:`n  $resultsDir\${Batch}_demo.log`n  $resultsDir\${Batch}_ambient.log"
Write-Host "Figures saved under `results\` folder."
