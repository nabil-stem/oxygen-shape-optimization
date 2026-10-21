param([string]$ResumeRoot = '')
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $repository '.venv\Scripts\python.exe'
$configPath = Join-Path $repository 'configs\morphology_trajectories.yaml'
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*oxygen_shape.experiments.morphology_study*' -and
        $_.CommandLine.Contains($repository) }
if ($running) { throw 'A morphology study is already running for this workspace.' }
$launchName = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '_' +
    [guid]::NewGuid().ToString('N').Substring(0, 8)
$launchRoot = Join-Path (Join-Path $repository 'experiments\launches') $launchName
New-Item -ItemType Directory -Path $launchRoot -Force | Out-Null
$arguments = '-m oxygen_shape.experiments.morphology_study --config "' + $configPath + '"'
if ($ResumeRoot) {
    $resumePath = (Resolve-Path -LiteralPath $ResumeRoot).Path
    if (-not $resumePath.StartsWith((Join-Path $repository 'experiments') + '\')) {
        throw 'Resume directory must be an experiment inside this workspace.'
    }
    $arguments += ' --resume "' + $resumePath + '"'
}
$oldOMP = $env:OMP_NUM_THREADS
$oldOpenBLAS = $env:OPENBLAS_NUM_THREADS
$oldMKL = $env:MKL_NUM_THREADS
try {
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
        -WorkingDirectory $repository -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $launchRoot 'stdout.log') `
        -RedirectStandardError (Join-Path $launchRoot 'stderr.log')
    $process.PriorityClass = 'BelowNormal'
    @{ pid = $process.Id; config = $configPath; resume = $ResumeRoot;
       launched_utc = (Get-Date).ToUniversalTime().ToString('o');
       workers = 2; numerical_threads_per_worker = 1 } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchRoot 'launch.json') -Encoding UTF8
    Write-Output "LAUNCH_DIRECTORY=$launchRoot"
    Write-Output "PID=$($process.Id)"
}
finally {
    $env:OMP_NUM_THREADS = $oldOMP
    $env:OPENBLAS_NUM_THREADS = $oldOpenBLAS
    $env:MKL_NUM_THREADS = $oldMKL
}
