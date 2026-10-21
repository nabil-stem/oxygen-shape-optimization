param(
    [string]$SourceRoot = 'experiments/20261021T122626095890Z_morphology_98e4aafa',
    [int]$DriverId = 0,
    [string]$ResumeRoot = '',
    [string]$GateXml = 'tmp/paper_gate_20261021/tests.xml'
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$studyPath = (Resolve-Path -LiteralPath $SourceRoot).Path
$gatePath = (Resolve-Path -LiteralPath $GateXml).Path
$experimentsPath = Join-Path $projectPath 'experiments'
function Get-QueuedHash([string]$FilePath) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString(
            $algorithm.ComputeHash([System.IO.File]::ReadAllBytes($FilePath))
        )).Replace('-', '')
    } finally { $algorithm.Dispose() }
}
if (-not $studyPath.StartsWith($experimentsPath + '\')) {
    throw 'The reference study must be an experiment within this workspace.'
}
if ($ResumeRoot) {
    $ResumeRoot = (Resolve-Path -LiteralPath $ResumeRoot).Path
    if (-not $ResumeRoot.StartsWith($experimentsPath + '\')) {
        throw 'The resumed controls must be an experiment within this workspace.'
    }
}
[xml]$gate = Get-Content -LiteralPath $gatePath -Raw
$suites = @($gate.SelectNodes('//testsuite'))
$testCount = ($suites | ForEach-Object { [int]$_.tests } | Measure-Object -Sum).Sum
$failureCount = ($suites | ForEach-Object { [int]$_.failures + [int]$_.errors } | Measure-Object -Sum).Sum
if ($testCount -lt 306 -or $failureCount -ne 0) {
    throw 'This production extension requires the completed 306-case paper regression gate.'
}
$running = Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and $_.Name -in @('powershell.exe', 'python.exe') -and
        ($_.CommandLine -like '*-File*complete_classical_controls.ps1*' -or
         $_.CommandLine -like '*-m scripts.conventional_study*') -and
        $_.CommandLine.Contains($projectPath) }
if ($running) { throw 'Classical controls are already queued or running in this workspace.' }
$launchName = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_classical_' +
    [guid]::NewGuid().ToString('N').Substring(0, 6)
$launchPath = Join-Path (Join-Path $projectPath 'experiments/launches') $launchName
New-Item -ItemType Directory -Path $launchPath | Out-Null
$snapshotPath = Join-Path $launchPath 'sources'
New-Item -ItemType Directory -Path $snapshotPath | Out-Null
$queuedSources = @{}
foreach ($name in @(
    'scripts/conventional_search.py', 'scripts/conventional_study.py',
    'scripts/audit_conventional.py', 'scripts/audit_morphology_journals.py',
    'scripts/postprocess_morphology_bounds.py', 'scripts/flux_energy_bounds.py',
    'scripts/complete_classical_controls.ps1', 'scripts/start_classical_controls.ps1',
    'docs/classical_control_protocol.md'
)) {
    $filePath = Join-Path $projectPath $name
    $queuedSources[$name] = Get-QueuedHash $filePath
    Copy-Item -LiteralPath $filePath -Destination (Join-Path $snapshotPath (Split-Path $filePath -Leaf))
}
$queuedSources | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchPath 'queued_sources.json') -Encoding UTF8
Copy-Item -LiteralPath $gatePath -Destination (Join-Path $launchPath 'tests.xml')
$workerScript = Join-Path $PSScriptRoot 'complete_classical_controls.ps1'
$runArguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $workerScript +
    '" -SourceRoot "' + $studyPath + '" -LaunchRoot "' + $launchPath +
    '" -DriverId ' + $DriverId
if ($ResumeRoot) { $runArguments += ' -ResumeRoot "' + $ResumeRoot + '"' }
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $runArguments `
    -WorkingDirectory $projectPath -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $launchPath 'stdout.log') `
    -RedirectStandardError (Join-Path $launchPath 'stderr.log')
$process.PriorityClass = 'BelowNormal'
@{ pid=$process.Id; source=$studyPath; resume=$ResumeRoot;
    launched_utc=[DateTime]::UtcNow.ToString('o'); workers=2;
    numerical_threads_per_worker=1; parent_overlapped_validation=$true;
    regression_cases=$testCount; gate_sha256=(Get-QueuedHash $gatePath) } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchPath 'launch.json') -Encoding UTF8
Write-Output "LAUNCH_DIRECTORY=$launchPath"
Write-Output "PID=$($process.Id)"
