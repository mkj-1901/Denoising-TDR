<#
.SYNOPSIS
Run the demo on several small batches and produce a CSV summary of PSNR results.
Extracts PSNR for every restoration rule from both the Gaussian and Ambient-Mean demos.
#>

$repoRoot   = (Get-Location).Path
$resultsDir = Join-Path $repoRoot "results"
if (!(Test-Path $resultsDir)) { New-Item -ItemType Directory -Path $resultsDir | Out-Null }

$batches = @(
    "clean",
    "denoising_gaussian001",
    "denoising_poisson_b1000b"
)

$csvPath = Join-Path $resultsDir "metrics_summary.csv"
"Batch,Method,Rule,PSNR_dB,BoundViolations" | Out-File -FilePath $csvPath -Encoding utf8

# The rules we want to extract from the logs
$rules = @("noisy y", "x* (R=inf)", "TDR-Init", "Ambient-Obs", "Ambient-Clean")

foreach ($batch in $batches) {
    Write-Host "`n=== Processing batch: $batch ==="
    .\run_batch.ps1 -Batch $batch

    foreach ($method in @("demo", "ambient")) {
        $logFile = Join-Path $resultsDir "${batch}_${method}.log"
        if (!(Test-Path $logFile)) {
            Write-Host "  WARNING: $logFile not found, skipping."
            continue
        }
        $logContent = Get-Content $logFile
        $methodLabel = if ($method -eq "demo") { "Gaussian" } else { "AmbientMean" }

        foreach ($rule in $rules) {
            # Find lines that contain this rule label AND a PSNR value
            $escapedRule = [regex]::Escape($rule)
            $matchLines = $logContent | Where-Object { $_ -match $escapedRule -and $_ -match 'true PSNR' }
            foreach ($ml in $matchLines) {
                $psnrMatch = [regex]::Match($ml, 'true PSNR = ([\d.]+) dB')
                $violMatch = [regex]::Match($ml, 'bound violations = \s*([\d.]+) %')
                $psnrVal = if ($psnrMatch.Success) { $psnrMatch.Groups[1].Value } else { "N/A" }
                $violVal = if ($violMatch.Success) { $violMatch.Groups[1].Value } else { "N/A" }
                "${batch},${methodLabel},${rule},${psnrVal},${violVal}" | Out-File -FilePath $csvPath -Append -Encoding utf8
            }
        }
    }
}

Write-Host "`n=== All batches processed ==="
Write-Host "CSV summary: $csvPath"
Write-Host "Figures and logs are under: $resultsDir"
