param(
    [string]$RepoRoot = '',
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [int]$Runs = 5,
    [int]$StartupTimeout = 30,
    [string]$Label = '',
    [string]$GitRef = '',
    [string]$ServerLog = '',
    [switch]$SkipDbBuild,
    [switch]$Headed
)

$ErrorActionPreference = 'Stop'
$root = if ($RepoRoot) { (Resolve-Path $RepoRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

$args = @(
  'run', '-n', 'dashmat', 'python',
  (Join-Path $root 'tools\playwright\portopt_import_harness.py'),
  '--repo-root', $root,
  '--base-url', $BaseUrl,
  '--runs', $Runs.ToString(),
  '--startup-timeout', $StartupTimeout.ToString(),
  '--label', $Label
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
