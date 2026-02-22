$ErrorActionPreference = 'Stop'
$session = 'reg_sweep3'
$root = 'C:\Git\DashMat'
$outDir = Join-Path $root 'output\playwright\regression'
$failDir = Join-Path $outDir 'failures'
$helperPath = (Join-Path $root 'tools\playwright\sweep_helper.js').Replace('\', '/')
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $failDir | Out-Null

function Pw([string[]]$CliArgs) {
  $cmd = @('--yes','--package','@playwright/cli','playwright-cli','-s',$session) + $CliArgs
  $out = & npx @cmd 2>&1 | Out-String
  return $out
}

function Extract-ResultObject([string]$out) {
  $m = [regex]::Match($out, '### Result\s*\r?\n(?<res>.*?)\r?\n### Ran', [System.Text.RegularExpressions.RegexOptions]::Singleline)
  if (-not $m.Success) { return $null }
  $res = $m.Groups['res'].Value.Trim()
  if (-not $res.StartsWith('{')) { return $null }
  try { return ($res | ConvertFrom-Json) } catch { return $null }
}

function Capture-Failure([string]$id) {
  $safe = $id -replace '[^A-Za-z0-9_-]', '_'
  $path = "output/playwright/regression/failures/$safe.png"
  $code = "async () => { await page.screenshot({ path: '$path', fullPage: true }); return { ok: true, path: '$path' }; }"
  try { [void](Pw @('run-code', $code)) } catch {}
  return $path
}

$results = @()

function Add-Case([string]$id, [string]$name, [string]$code, [scriptblock]$validator = $null) {
  Write-Host "Running $id - $name"
  $out = Pw @('run-code', $code)
  if ($out -match '### Error') {
    $shot = Capture-Failure $id
    $script:results += [pscustomobject]@{ id=$id; name=$name; status='FAIL'; error='Playwright CLI Error'; screenshot=$shot }
    return
  }
  $obj = Extract-ResultObject $out
  if ($null -eq $obj) {
    $shot = Capture-Failure $id
    $script:results += [pscustomobject]@{ id=$id; name=$name; status='FAIL'; error='Could not parse result object'; screenshot=$shot }
    return
  }

  $ok = $false
  try {
    if ($null -ne $validator) {
      $ok = [bool](& $validator $obj)
    } else {
      $ok = [bool]$obj.ok
    }
  } catch { $ok = $false }

  if ($ok) {
    $script:results += [pscustomobject]@{ id=$id; name=$name; status='PASS'; detail=$obj }
  } else {
    $shot = Capture-Failure $id
    $script:results += [pscustomobject]@{ id=$id; name=$name; status='FAIL'; detail=$obj; screenshot=$shot }
  }
}

# Clean sessions and open browser
[void](Pw @('kill-all'))
[void](Pw @('open','http://127.0.0.1:8050/regression','--headed'))

$daily = 'C:/Git/DashMat/sample_data/benchmark_returns/benchmark_daily_returns_2020_2025.xlsx'
$monthly = 'C:/Git/DashMat/sample_data/benchmark_returns/benchmark_monthly_returns_2020_2025.xlsx'

Add-Case 'C01' 'Upload daily sample and show main' "async () => { const input = page.locator('#reg-upload-data input[type=file]'); await input.setInputFiles('$daily'); await page.waitForTimeout(2600); const main = await page.locator('#reg-main-container').isVisible(); const modal = await page.locator('#reg-modal-ok-button').isVisible().catch(() => false); return { ok: main, main, modal }; }"

Add-Case 'C02' 'Series modal opens and can close' "async () => { const okBtn = page.locator('#reg-modal-ok-button'); const opened = await okBtn.isVisible().catch(() => false); const cancel = page.locator('#reg-modal-cancel-button'); if (await cancel.isVisible().catch(() => false)) { await cancel.click(); await page.waitForTimeout(500); } return { ok: opened, opened }; }"

# Initialize in-page helper
Add-Case 'C03' 'Initialize helper runtime' "async () => { await page.addScriptTag({ path: '$helperPath' }); const ok = await page.evaluate(() => !!window.__sweep); return { ok }; }"

# Model/run matrix
Add-Case 'C04' 'OLS full succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'ols', windowType: 'full', forceZero: false }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C05' 'Ridge succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'ridge', alpha: 1.0 }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C06' 'Lasso succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'lasso', alpha: 0.8 }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C07' 'Elastic Net succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'elastic_net', alpha: 0.8, l1Ratio: 0.4 }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C08' 'Constrained OLS succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'constrained_ols' }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C09' 'Style Analysis succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'style_analysis' }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C10' 'OLS intercept-only succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',[]); await window.__sweep.setRun({ model: 'ols', forceZero: false, arimaP: 0, arimaD: 0, arimaQ: 0, garchP: 0, garchQ: 0 }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C11' 'Intercept-only + force-zero errors' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',[]); await window.__sweep.setRun({ model: 'ols', forceZero: true, arimaP: 0, arimaD: 0, arimaQ: 0, garchP: 0, garchQ: 0 }); const msg = await window.__sweep.run(); return { ok: msg.toLowerCase().includes('error'), msg }; }); }"
Add-Case 'C12' 'ARIMA/GARCH no-X succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',[]); await window.__sweep.setRun({ model: 'ols', forceZero: false, arimaP: 1, arimaD: 0, arimaQ: 1, garchP: 1, garchQ: 1 }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"
Add-Case 'C13' 'Rolling window succeeds' "async () => { return await page.evaluate(async () => { await window.__sweep.setSeries('SPX',['RMID','R2000','EAFE']); await window.__sweep.setRun({ model: 'ols', windowType: 'rolling', windowSize: 252, optStep: 21, optStepUnit: 'periods' }); const msg = await window.__sweep.run(); return { ok: msg.startsWith('✓'), msg }; }); }"

# Output tabs and modes
Add-Case 'C14' 'ANOVA latest window default' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('ANOVA'); const v = document.querySelector('#reg-anova-window-select')?.value || ''; return { ok: !!v && !v.includes('Window 1:'), value: v }; }); }"
Add-Case 'C15' 'Rolling Summary table/chart render' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Rolling Summary'); window.__sweep.setProps({ 'reg-rolling-summary-chart-switch': { value: 'table' }, 'reg-rolling-summary-detail-switch': { value: 'basic' } }); await window.__sweep.sleep(500); const t = window.__sweep.hasGrid('#reg-rolling-content'); window.__sweep.setProps({ 'reg-rolling-summary-chart-switch': { value: 'chart' }, 'reg-rolling-summary-detail-switch': { value: 'advanced' } }); await window.__sweep.sleep(700); const c = window.__sweep.hasGraph('#reg-rolling-content'); return { ok: t && c, table: t, chart: c }; }); }"
Add-Case 'C16' 'Weights table has only coefficient-style columns' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Weights'); window.__sweep.setProps({ 'reg-weights-chart-switch': { value: 'table' } }); await window.__sweep.sleep(700); const headers = window.__sweep.headers('#reg-weights-content'); const ok = headers.includes('Window') && headers.includes('Date') && !headers.some(h => h.startsWith('ARIMA_') || h.startsWith('GARCH_')); return { ok, headers }; }); }"
Add-Case 'C17' 'Statistics and Returns grids render' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Statistics'); const s = window.__sweep.hasGrid('#reg-statistics-content'); await window.__sweep.tab('Returns'); const r = window.__sweep.hasGrid('#reg-returns-content'); return { ok: s && r, statistics: s, returns: r }; }); }"
Add-Case 'C18' 'Growth and Drawdown chart/table render' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Growth of `$1'); window.__sweep.setProps({ 'reg-growth-chart-switch': { value: 'chart' } }); await window.__sweep.sleep(600); const gc = window.__sweep.hasGraph('#reg-growth-content'); window.__sweep.setProps({ 'reg-growth-chart-switch': { value: 'table' } }); await window.__sweep.sleep(600); const gt = window.__sweep.hasGrid('#reg-growth-content'); await window.__sweep.tab('Drawdown'); window.__sweep.setProps({ 'reg-drawdown-chart-switch': { value: 'chart' } }); await window.__sweep.sleep(600); const dc = window.__sweep.hasGraph('#reg-drawdown-content'); window.__sweep.setProps({ 'reg-drawdown-chart-switch': { value: 'table' } }); await window.__sweep.sleep(600); const dt = window.__sweep.hasGrid('#reg-drawdown-content'); return { ok: gc && gt && dc && dt, gc, gt, dc, dt }; }); }"
Add-Case 'C19' 'Rolling tab chart/table render' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Rolling'); window.__sweep.setProps({ 'reg-rolling-metric-select': { value: 'volatility' }, 'reg-rolling-window-select': { value: '1y' }, 'reg-rolling-return-type-select': { value: 'annualized' }, 'reg-rolling-chart-switch': { value: 'chart' } }); await window.__sweep.sleep(900); const c = window.__sweep.hasGraph('#reg-rolling-returns-content'); window.__sweep.setProps({ 'reg-rolling-chart-switch': { value: 'table' } }); await window.__sweep.sleep(700); const t = window.__sweep.hasGrid('#reg-rolling-returns-content'); return { ok: c && t, chart: c, table: t }; }); }"
Add-Case 'C20' 'Scatter modes render, trend line present for residual vs predicted' "async () => { return await page.evaluate(async () => { await window.__sweep.tab('Scatter'); window.__sweep.setProps({ 'reg-scatter-mode-select': { value: 'residual_vs_predicted' } }); await window.__sweep.sleep(900); const tr = window.__sweep.scatterTraceNames(); const a = window.__sweep.hasGraph('#reg-scatter-content') && tr.includes('Trend Line'); window.__sweep.setProps({ 'reg-scatter-mode-select': { value: 'actual_vs_predicted' } }); await window.__sweep.sleep(900); const b = window.__sweep.hasGraph('#reg-scatter-content'); window.__sweep.setProps({ 'reg-scatter-mode-select': { value: 'actual_vs_x' }, 'reg-scatter-x-select': { value: 'RMID' } }); await window.__sweep.sleep(900); const c = window.__sweep.hasGraph('#reg-scatter-content'); window.__sweep.setProps({ 'reg-scatter-mode-select': { value: 'predicted_vs_x' }, 'reg-scatter-x-select': { value: 'EAFE' } }); await window.__sweep.sleep(900); const d = window.__sweep.hasGraph('#reg-scatter-content'); return { ok: a && b && c && d, trendTraces: tr, actualPred: b, actualX: c, predX: d }; }); }"
Add-Case 'C21' 'Delete result changes current selection' "async () => { return await page.evaluate(async () => { const x = await window.__sweep.deleteResult(); return { ok: x.before !== x.after, ...x }; }); }"

# Reset and monthly upload check
Add-Case 'C22' 'Monthly upload periodicity is monthly' "async () => { await page.evaluate(() => { try { sessionStorage.clear(); } catch (_) {} }); await page.goto('http://127.0.0.1:8050/regression'); await page.waitForLoadState('networkidle'); await page.waitForTimeout(700); const input = page.locator('#reg-upload-data input[type=file]'); await input.setInputFiles('$monthly'); await page.waitForTimeout(2800); const main = await page.locator('#reg-main-container').isVisible().catch(() => false); const cancel = page.locator('#reg-modal-cancel-button'); if (await cancel.isVisible().catch(() => false)) { await cancel.click(); await page.waitForTimeout(400); } const v = await page.locator('#reg-periodicity-select').inputValue(); return { ok: main && String(v).toLowerCase().includes('monthly'), main, periodicity: v }; }"

# Close session
[void](Pw @('close'))

$summary = [pscustomobject]@{
  total = $results.Count
  passed = ($results | Where-Object { $_.status -eq 'PASS' }).Count
  failed = ($results | Where-Object { $_.status -eq 'FAIL' }).Count
  cases = $results
}

$summaryPath = Join-Path $outDir 'summary.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath

Write-Output ("SUMMARY_PATH=" + $summaryPath)
Write-Output ("TOTAL=" + $summary.total)
Write-Output ("PASSED=" + $summary.passed)
Write-Output ("FAILED=" + $summary.failed)

