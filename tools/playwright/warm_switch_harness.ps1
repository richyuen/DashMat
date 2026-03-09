param(
    [string]$BaseUrl = 'http://127.0.0.1:8050',
    [string]$SampleFile = 'C:/Git/DashMat/sample_data/benchmark_returns/benchmark_daily_returns_2020_2025.xlsx',
    [int]$Runs = 5,
    [string]$Session = 'warm_switch_harness',
    [switch]$Headed
)

$ErrorActionPreference = 'Stop'

$root = 'C:\Git\DashMat'
$outDir = Join-Path $root 'output\playwright'
$failDir = Join-Path $outDir 'failures'

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $failDir | Out-Null

function Pw([string[]]$CliArgs) {
  $cmd = @('--yes', '--package', '@playwright/cli', 'playwright-cli', '-s', $Session) + $CliArgs
  $out = & npx @cmd 2>&1 | Out-String
  return $out
}

function Extract-ResultObject([string]$out) {
  $m = [regex]::Match(
    $out,
    '### Result\s*\r?\n(?<res>.*?)\r?\n### Ran',
    [System.Text.RegularExpressions.RegexOptions]::Singleline
  )
  if (-not $m.Success) { return $null }
  $res = $m.Groups['res'].Value.Trim()
  if (-not $res.StartsWith('{')) { return $null }
  try { return ($res | ConvertFrom-Json -Depth 20) } catch { return $null }
}

function Capture-Failure([string]$name) {
  $safe = $name -replace '[^A-Za-z0-9_-]', '_'
  $path = "output/playwright/failures/$safe.png"
  $code = "async () => { await page.screenshot({ path: '$path', fullPage: true }); return { ok: true, path: '$path' }; }"
  try { [void](Pw @('run-code', $code)) } catch {}
  return (Join-Path $root $path.Replace('/', '\'))
}

[void](Pw @('kill-all'))
$openArgs = @('open', ($BaseUrl + '/analyticstool'))
if ($Headed) {
  $openArgs += '--headed'
}
[void](Pw $openArgs)

$jsBaseUrl = $BaseUrl.Replace('\', '\\')
$jsSampleFile = $SampleFile.Replace('\', '/')

$code = @"
async () => {
  const baseUrl = '$jsBaseUrl';
  const sampleFile = '$jsSampleFile';
  const runs = $Runs;
  const pages = {
    analytics: { path: '/analyticstool', shell: '#at-main-app-container', ready: '#at-periodicity-select' },
    portopt: { path: '/portopt', shell: '#po-main-container', ready: '#po-periodicity-select' },
    regression: { path: '/regression', shell: '#reg-main-container', ready: '#reg-periodicity-select' },
  };
  const consoleMessages = [];
  const onConsole = (msg) => {
    if (consoleMessages.length >= 120) return;
    const type = msg.type();
    if (type === 'error' || type === 'warning') {
      consoleMessages.push({ type, text: msg.text() });
    }
  };
  const onPageError = (err) => {
    if (consoleMessages.length >= 120) return;
    consoleMessages.push({ type: 'pageerror', text: String(err) });
  };
  page.on('console', onConsole);
  page.on('pageerror', onPageError);

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  };

  async function waitVisible(selector, timeout = 30000) {
    await page.waitForFunction(
      (sel) => {
        const el = document.querySelector(sel);
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
      },
      selector,
      { timeout }
    );
  }

  async function waitReady(selector, timeout = 30000) {
    await page.waitForFunction(
      (sel) => {
        const el = document.querySelector(sel);
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        const visible = style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
        if (!visible) return false;
        const selfDisabled =
          !!el.disabled ||
          el.getAttribute('disabled') !== null ||
          el.getAttribute('aria-disabled') === 'true';
        const ancestorDisabled = !!el.closest('[aria-disabled="true"], [disabled]');
        return !(selfDisabled || ancestorDisabled);
      },
      selector,
      { timeout }
    );
  }

  async function measure(name) {
    const cfg = pages[name];
    const start = Date.now();
    await page.evaluate((targetPath) => {
      window.location.pathname = targetPath;
    }, cfg.path);
    await page.waitForFunction((targetPath) => window.location.pathname === targetPath, cfg.path, { timeout: 30000 });
    await waitVisible(cfg.shell);
    const shellMs = Date.now() - start;
    await waitReady(cfg.ready);
    const readyMs = Date.now() - start;
    return { shellMs, readyMs };
  }

  await page.goto(baseUrl + pages.analytics.path);
  await page.locator('#at-upload-data input[type=file]').setInputFiles(sampleFile);
  await page.waitForTimeout(800);

  const sheetModal = page.locator('#at-sheet-select-modal');
  if (await sheetModal.isVisible().catch(() => false)) {
    const importAll = page.locator('#at-sheet-select-import-all-button');
    if (await importAll.isVisible().catch(() => false)) {
      await importAll.click();
    }
  }

  await waitVisible(pages.analytics.shell);
  await waitReady(pages.analytics.ready);

  await measure('portopt');
  await measure('regression');

  const results = {
    analytics: { runs: 0, shellMs: [], readyMs: [] },
    portopt: { runs: 0, shellMs: [], readyMs: [] },
    regression: { runs: 0, shellMs: [], readyMs: [] },
  };

  for (let i = 0; i < runs; i += 1) {
    for (const name of ['analytics', 'portopt', 'regression']) {
      const metrics = await measure(name);
      results[name].runs += 1;
      results[name].shellMs.push(metrics.shellMs);
      results[name].readyMs.push(metrics.readyMs);
    }
  }

  for (const [name, data] of Object.entries(results)) {
    data.shellMedian = median(data.shellMs);
    data.readyMedian = median(data.readyMs);
  }

  return { ok: true, baseUrl, sampleFile, runs, results, consoleMessages };
}
"@
# Windows + playwright-cli are unreliable with multiline run-code payloads.
# Flatten the script before invocation so the CLI receives one argument cleanly.
$code = $code -replace "(`r`n|`n|`r)", ' '

$raw = Pw @('run-code', $code)
$result = Extract-ResultObject $raw
if ($null -eq $result -or -not $result.ok) {
  $shot = Capture-Failure 'warm_switch_harness'
  $rawPath = Join-Path $outDir 'warm_switch_last_raw.txt'
  Set-Content -Path $rawPath -Value $raw -Encoding utf8
  Write-Output ("RAW_PATH=" + $rawPath)
  throw "Failed to parse warm-switch result object. Screenshot: $shot"
}

$stamp = Get-Date -Format 'yyyy-MM-ddTHH-mm-ss'
$outPath = Join-Path $outDir ("warm_switch_timing_" + $stamp + '.json')
$payload = [ordered]@{
  timestamp = (Get-Date).ToString('o')
  baseUrl = $result.baseUrl
  sampleFile = $result.sampleFile
  runs = $result.runs
  analytics = $result.results.analytics
  portopt = $result.results.portopt
  regression = $result.results.regression
  consoleMessages = $result.consoleMessages
}
$payload | ConvertTo-Json -Depth 20 | Set-Content -Path $outPath -Encoding utf8

[void](Pw @('close'))

Write-Output ("OUT_PATH=" + $outPath)
Write-Output ("ANALYTICS=" + (($result.results.analytics | ConvertTo-Json -Compress)))
Write-Output ("PORTOPT=" + (($result.results.portopt | ConvertTo-Json -Compress)))
Write-Output ("REGRESSION=" + (($result.results.regression | ConvertTo-Json -Compress)))
