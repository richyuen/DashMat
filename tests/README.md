# DashMat Automated Tests

This directory is the single source of truth for automated validation.

## Scope

- `tests/unit/`: utility-layer behavior and deterministic calculation logic.
- `tests/callbacks/`: callback-level contracts for `pages/analyticstool.py` and `pages/portopt.py`.
- `tests/scripts/`: script/flow verification for optimization checks, benchmark tooling, and data-generation helpers.

## Run

```bash
# Install dev-only test dependencies
conda run -n dashmat python -m pip install -r requirements-dev.txt

# Run full suite
conda run -n dashmat python -m pytest -q tests

# Optional callback-level upload smoke checks
conda run -n dashmat python tools/smoke_upload_flow.py --file C:\Git\SampleMstar.xlsx --page both --mode both
conda run -n dashmat python tools/smoke_upload_flow.py --file C:\Git\SampleMstarMulti.xlsx --page both --mode both

# Optional upload/date-range latency benchmark
conda run -n dashmat python tools/benchmark_upload_date_range.py
```

## Coverage Gate

Coverage is enforced at `>=85%` for core deterministic modules configured in `pytest.ini`:

- `cache_config`
- `utils.serialization`
- `utils.perf_timing`
- `utils.parsing`
- `tools.data.generate_test_data`

## Notes

- Tests are isolated: no live network or production DB dependencies.
- Page callback tests import `app` first (required before importing `pages/*` modules due `register_page`).
- Legacy root tests were migrated into this folder; use pytest instead of direct script execution.
