param(
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [Parameter(Mandatory = $true)][int]$DriverId
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$studyRoot = (Resolve-Path -LiteralPath $RunRoot).Path
$reviewRoot = Join-Path $studyRoot 'postprocess/review'
New-Item -ItemType Directory -Path $reviewRoot -Force | Out-Null
$reviewStatus = Join-Path $reviewRoot 'status.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Set-Location -LiteralPath $projectRoot

function Write-ReviewStatus([string]$Phase, [string]$Status, [string]$Detail = '') {
    $state = @{
        phase = $Phase
        status = $Status
        detail = $Detail
        utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json
    [System.IO.File]::WriteAllText($reviewStatus, $state, $utf8)
}

try {
    Write-ReviewStatus 'waiting_for_main_study' 'running'
    while ($true) {
        $studyStatus = Get-Content -LiteralPath (Join-Path $studyRoot 'status.json') -Raw |
            ConvertFrom-Json
        if ($studyStatus.phase -eq 'finished') { break }
        if (-not (Get-Process -Id $DriverId -ErrorAction SilentlyContinue)) {
            throw 'The main driver stopped before finishing; inspect its logs and resume it first.'
        }
        Start-Sleep -Seconds 20
    }
    $python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    foreach ($name in @(
        'postprocess_morphology_bounds.py',
        'audit_morphology_journals.py',
        'collar_ablation.py',
        'source_corner_validation.py'
    )) {
        Write-ReviewStatus $name 'running'
        Write-Output ("Starting {0} at {1}" -f $name, [DateTime]::UtcNow.ToString('o'))
        & $python (Join-Path $PSScriptRoot $name) $studyRoot
        if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
    }
    Write-ReviewStatus 'finished' 'complete'
    Write-Output 'All independent morphology postprocessing completed.'
} catch {
    Write-ReviewStatus 'failed' 'error' $_.Exception.Message
    throw
}
