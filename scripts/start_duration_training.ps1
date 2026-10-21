param(
    [string]$Config = "configs/learning_duration.yaml",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$durationRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$durationPython = Join-Path $durationRoot ".venv/Scripts/python.exe"
$durationConfig = (Resolve-Path (Join-Path $durationRoot $Config)).Path
$existingDuration = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*-m oxygen_shape.experiments.learning_duration*" }
if ($existingDuration) {
    throw "A duration-training process is already running; refusing a duplicate launch."
}
$durationStamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$durationLaunch = Join-Path $durationRoot ("experiments/launches/" + $durationStamp + "_" + [Guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $durationLaunch | Out-Null
$durationArguments = '-u -m oxygen_shape.experiments.learning_duration --config "' + $durationConfig + '"'
if ($Resume) {
    $durationResume = (Resolve-Path $Resume).Path
    $durationArguments += ' --resume "' + $durationResume + '"'
}
$durationProcess = Start-Process -FilePath $durationPython -ArgumentList $durationArguments `
    -WorkingDirectory $durationRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $durationLaunch "stdout.log") `
    -RedirectStandardError (Join-Path $durationLaunch "stderr.log") -PassThru
$durationMetadata = [ordered]@{
    process_id = $durationProcess.Id
    started_at_utc = [DateTime]::UtcNow.ToString("o")
    config = $durationConfig
    config_sha256 = (Get-FileHash -LiteralPath $durationConfig -Algorithm SHA256).Hash.ToLowerInvariant()
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    resume = $Resume
    stdout = (Join-Path $durationLaunch "stdout.log")
    stderr = (Join-Path $durationLaunch "stderr.log")
}
$durationMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $durationLaunch "launch.json") -Encoding utf8
$durationMetadata | ConvertTo-Json
