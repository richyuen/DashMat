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
$runner = Join-Path $root 'tools\playwright\start_timed_server_runner.py'
$condaExe = (Get-Command conda.exe).Source

$previousTimingEnabled = $env:DASHMAT_TIMING_ENABLED
$previousTimingMinMs = $env:DASHMAT_TIMING_MIN_MS
$previousCompression = $env:DASHMAT_ENABLE_COMPRESSION

$env:DASHMAT_TIMING_ENABLED = '1'
$env:DASHMAT_TIMING_MIN_MS = $TimingMinMs
if ($Compression) {
    $env:DASHMAT_ENABLE_COMPRESSION = '1'
} else {
    Remove-Item Env:DASHMAT_ENABLE_COMPRESSION -ErrorAction SilentlyContinue
}

try {
    $proc = Start-Process -FilePath $condaExe `
        -ArgumentList @(
            'run',
            '--no-capture-output',
            '-n',
            'dashmat',
            'python',
            '-u',
            $runner,
            '--port',
            $Port.ToString()
        ) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
}
finally {
    if ($null -ne $previousTimingEnabled) {
        $env:DASHMAT_TIMING_ENABLED = $previousTimingEnabled
    } else {
        Remove-Item Env:DASHMAT_TIMING_ENABLED -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousTimingMinMs) {
        $env:DASHMAT_TIMING_MIN_MS = $previousTimingMinMs
    } else {
        Remove-Item Env:DASHMAT_TIMING_MIN_MS -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousCompression) {
        $env:DASHMAT_ENABLE_COMPRESSION = $previousCompression
    } else {
        Remove-Item Env:DASHMAT_ENABLE_COMPRESSION -ErrorAction SilentlyContinue
    }
}

Write-Output ("PID=" + $proc.Id)
Write-Output ("STDOUT=" + $stdout)
Write-Output ("STDERR=" + $stderr)
Write-Output ("RUNNER=" + $runner)
