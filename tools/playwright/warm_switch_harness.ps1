param(
    [string]$RepoRoot = '',
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [string[]]$DbSeries = @('SPX_TRIndex', 'R2000_TRIndex', 'EAFE_TRIndex', 'BCTBill13_TRIndex'),
    [string[]]$Pages = @('analytics', 'portopt', 'regression'),
    [int]$Runs = 5,
    [int]$StartupTimeout = 30,
    [string]$Label = '',
    [string]$GitRef = '',
    [string]$ServerLog = '',
    [string]$PortoptRestoreTab = 'weight',
    [switch]$PortoptEntryOnly,
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
  '--portopt-restore-tab', $PortoptRestoreTab
)

if ($GitRef) {
  $args += '--git-ref'
  $args += $GitRef
}

if ($Headed) {
  $args += '--headed'
}
if ($PortoptEntryOnly) {
  $args += '--portopt-entry-only'
}
if ($SkipDbBuild) {
  $args += '--skip-db-build'
}
if ($DbSeries -and $DbSeries.Count -gt 0) {
  $args += '--db-series'
  $args += $DbSeries
}
if ($Pages -and $Pages.Count -gt 0) {
  $args += '--pages'
  $args += $Pages
}
if ($ServerLog) {
  $args += '--server-log'
  $args += $ServerLog
}

& conda @args
exit $LASTEXITCODE
