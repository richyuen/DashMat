param(
    [string]$RepoRoot = '',
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [int]$Runs = 3,
    [int]$StartupTimeout = 30,
    [string]$Label = '',
    [string]$GitRef = '',
    [ValidateSet('consume_only', 'run_and_consume')]
    [string]$Mode = 'consume_only',
    [switch]$Headed
)

$ErrorActionPreference = 'Stop'
$root = if ($RepoRoot) { (Resolve-Path $RepoRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

$args = @(
  'run', '-n', 'dashmat', 'python',
  (Join-Path $root 'tools\playwright\result_flow_harness.py'),
  '--repo-root', $root,
  '--base-url', $BaseUrl,
  '--runs', $Runs.ToString(),
  '--startup-timeout', $StartupTimeout.ToString(),
  '--label', $Label,
  '--git-ref', $GitRef,
  '--mode', $Mode
)

if ($Headed) {
  $args += '--headed'
}

& conda @args
exit $LASTEXITCODE
