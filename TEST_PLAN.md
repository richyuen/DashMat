# DashMat Test Plan

This plan defines the expected validation flow for production changes.

## 1. Environment Setup

```bash
conda create -n dashmat python=3.11 -y
conda activate dashmat
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

For non-interactive shells:

```bash
conda run -n dashmat python -m pip install -r requirements.txt
conda run -n dashmat python -m pip install -r requirements-dev.txt
```

## 2. Automated Validation (Required)

Run the full suite:

```bash
conda run -n dashmat python -m pytest -q tests
```

The canonical automated test scope and coverage rules are maintained in `tests/README.md`.

## 3. Manual Validation (Targeted)

Run manual checks only for flows touched by the change.

### 3.1 Analytics Tool (`/analyticstool`)

- Upload/parse returns file (CSV/XLS/XLSX, percent cells).
- Validate `Add portfolios (peer)`, `Add portfolios (index)`, and `Add portfolios (other)` flows from menu/welcome buttons.
- For `other`, validate AltTS benchmark mapping (`Portfolio` + `BenchRet` -> `<Portfolio>_BM`).
- Confirm periodicity controls and date range filtering.
- Check `Statistics`, `Returns`, `Growth`, `Drawdown`, and `Correlation` tabs for callback errors.
- Validate Excel export succeeds from File menu.

### 3.2 Portfolio Optimization (`/portopt`)

- Run at least one optimization model with realistic inputs.
- Validate `Add portfolios (peer)`, `Add portfolios (index)`, and `Add portfolios (other)` flows.
- Confirm completion modal and saved portfolio visibility.
- Validate `Weights`, `Attribution`, `Statistics`, `Returns`, `Risk`, and `Turnover` views.
- Validate portfolio delete/sync behavior against Analytics data.

## 4. Performance/Instrumentation (As Needed)

- Run callback benchmark harness when callback latency changes are introduced:

```bash
conda run -n dashmat python tools/benchmark_callback_latency.py
```

- If timing instrumentation is enabled (`DASHMAT_TIMING_ENABLED=1`), review callback timing logs for regressions.
