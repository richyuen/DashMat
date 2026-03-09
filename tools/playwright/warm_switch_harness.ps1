param(
    [string]$RepoRoot = '',
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [string[]]$DbSeries = @('SPX_TRIndex', 'R2000_TRIndex', 'EAFE_TRIndex', 'BCTBill13_TRIndex'),
    [int]$Runs = 5,
    [int]$StartupTimeout = 30,
    [string]$Label = '',
    [string]$GitRef = '',
    [switch]$SkipDbBuild,
    [switch]$Headed
)

$ErrorActionPreference = 'Stop'
$root = if ($RepoRoot) { (Resolve-Path $RepoRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

$args = @(
  'run', '-n', 'dashmat', 'python',
  (Join-Path $root 'tools\playwright\warm_switch_harness.py'),
  '--repo-root', $root,
  '--base-url', $BaseUrl,
  '--runs', $Runs.ToString(),
  '--startup-timeout', $StartupTimeout.ToString(),
  '--label', $Label,
  '--git-ref', $GitRef
)

if ($Headed) {
  $args += '--headed'
}
if ($SkipDbBuild) {
  $args += '--skip-db-build'
}
if ($DbSeries -and $DbSeries.Count -gt 0) {
  $args += '--db-series'
  $args += $DbSeries
}

& conda @args
exit $LASTEXITCODE
