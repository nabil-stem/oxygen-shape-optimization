param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$LaunchRoot,
    [int]$DriverId = 0,
    [string]$ResumeRoot = ''
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$studyPath = (Resolve-Path -LiteralPath $SourceRoot).Path
$launchPath = (Resolve-Path -LiteralPath $LaunchRoot).Path
$statusPath = Join-Path $launchPath 'status.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)
Set-Location -LiteralPath $projectPath
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'

function Write-ControlStatus([string]$Phase, [string]$Status, [string]$Detail = '') {
    $record = @{ phase=$Phase; status=$Status; detail=$Detail;
        utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json
    [System.IO.File]::WriteAllText($statusPath, $record, $utf8)
}

function Get-QueuedHash([string]$FilePath) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString(
            $algorithm.ComputeHash([System.IO.File]::ReadAllBytes($FilePath))
        )).Replace('-', '')
    } finally { $algorithm.Dispose() }
}

try {
    Write-ControlStatus 'waiting_for_completed_source' 'running'
    while ($true) {
        $state = Get-Content -LiteralPath (Join-Path $studyPath 'status.json') -Raw |
            ConvertFrom-Json
        if ($state.phase -eq 'finished') { break }
        if ($DriverId -eq 0 -or -not (Get-Process -Id $DriverId -ErrorAction SilentlyContinue)) {
            throw 'The source study is unfinished and its expected driver is absent.'
        }
        Start-Sleep -Seconds 20
    }
    $manifest = Get-Content -LiteralPath (Join-Path $launchPath 'queued_sources.json') -Raw |
        ConvertFrom-Json
    foreach ($entry in $manifest.PSObject.Properties) {
        $actual = Get-QueuedHash (Join-Path $projectPath $entry.Name)
        if ($actual -ne $entry.Value) { throw "Queued control source changed: $($entry.Name)" }
    }
    Write-ControlStatus 'classical_search_and_validation' 'running'
    $pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
    $runArguments = @('-m', 'scripts.conventional_study', $studyPath, '--workers', '2')
    if ($ResumeRoot) { $runArguments += @('--resume', (Resolve-Path -LiteralPath $ResumeRoot).Path) }
    & $pythonPath @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Classical controls failed with exit code $LASTEXITCODE" }
    Write-ControlStatus 'finished' 'complete'
} catch {
    Write-ControlStatus 'failed' 'error' $_.Exception.Message
    throw
}
