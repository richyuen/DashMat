param(
    [string]$RepoRoot = '',
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [int]$Runs = 5,
    [string]$Label = '',
    [string]$GitRef = '',
    [string]$ServerLog = '',
    [string]$Mode = 'imports',
    [int]$StartupTimeout = 30,
    [switch]$SkipDbBuild,
    [switch]$Headed
)

$ErrorActionPreference = 'Stop'
$root = if ($RepoRoot) { (Resolve-Path $RepoRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

$args = @(
  'run', '-n', 'dashmat', 'python',
  (Join-Path $root 'tools\playwright\at_statistics_harness.py'),
  '--repo-root', $root,
  '--base-url', $BaseUrl,
  '--runs', $Runs.ToString(),
  '--startup-timeout', $StartupTimeout.ToString(),
  '--label', $Label,
  '--mode', $Mode
)

if ($GitRef) {
  $args += '--git-ref'
  $args += $GitRef
}

if ($Headed) {
  $args += '--headed'
}
if ($SkipDbBuild) {
  $args += '--skip-db-build'
}
if ($ServerLog) {
  $args += '--server-log'
  $args += $ServerLog
}

& conda @args
exit $LASTEXITCODE
