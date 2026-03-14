param(
    [string]$RepoRoot = '',
    [int]$Port = 8050,
    [string]$LogDir = '',
    [string]$LogStem = '',
    [string]$TimingMinMs = '0',
    [switch]$Compression
)

$ErrorActionPreference = 'Stop'

$root = if ($RepoRoot) {
    (Resolve-Path $RepoRoot).Path
} else {
    (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

$resolvedLogDir = if ($LogDir) {
    $LogDir
} else {
    Join-Path $root 'output\playwright'
}
New-Item -ItemType Directory -Force -Path $resolvedLogDir | Out-Null

$stem = if ($LogStem) {
    $LogStem
} else {
    "server_p$Port"
}

$stdout = Join-Path $resolvedLogDir "${stem}_stdout.log"
$stderr = Join-Path $resolvedLogDir "${stem}_stderr.log"
$launcher = Join-Path $resolvedLogDir "${stem}_launch.cmd"

@"
@echo off
cd /d "$root"
set DASHMAT_TIMING_ENABLED=1
set DASHMAT_TIMING_MIN_MS=$TimingMinMs
set DASHMAT_ENABLE_COMPRESSION=
"@ | Set-Content -Path $launcher -Encoding ASCII

if ($Compression) {
    Add-Content -Path $launcher -Value 'set DASHMAT_ENABLE_COMPRESSION=1'
}

Add-Content -Path $launcher -Value "conda run --no-capture-output -n dashmat python -u -c ""import app; app.app.run(debug=False, port=$Port, use_reloader=False)"" 1> ""$stdout"" 2> ""$stderr"""

$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $launcher) -WorkingDirectory $root -WindowStyle Hidden -PassThru

Write-Output ("PID=" + $proc.Id)
Write-Output ("STDOUT=" + $stdout)
Write-Output ("STDERR=" + $stderr)
Write-Output ("LAUNCHER=" + $launcher)
