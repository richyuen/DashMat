# AGENTS.md

Instructions for coding agents working in `C:\Git\DashMat`.

## Objective

Build and maintain DashMat, a Dash-based market returns analytics app with two main pages:
- `pages/analyticstool.py` for data analysis workflows
- `pages/portopt.py` for portfolio optimization workflows

Prefer small, targeted changes that preserve existing behavior unless a behavior change is explicitly requested.

## Stack

- Python 3.11
- Dash 2.14+
- Dash Mantine Components
- Dash AG Grid
- pandas / scipy
- riskfolio-lib
- Flask-Caching

## Setup

```bash
conda create -n dashmat python=3.11 -y
conda activate dashmat
pip install -r requirements.txt
```

## Environment execution rule

- Always run project commands in the `dashmat` Conda environment.
- In non-interactive or tool-driven shells, prefer `conda run -n dashmat <command>` instead of relying on `conda activate`.

## Run

```bash
python app.py
python app.py --debug
# Non-interactive shell alternative:
conda run -n dashmat python app.py
conda run -n dashmat python app.py --debug
```

## Tests and checks

```bash
python test_factor_rp.py
python verify_linear_constraints.py
python generate_test_data.py
# Non-interactive shell alternative:
conda run -n dashmat python test_factor_rp.py
conda run -n dashmat python verify_linear_constraints.py
conda run -n dashmat python generate_test_data.py
```

If you change optimization logic, run both optimization-related checks.
If you change upload/parsing/statistics flows, also do a quick manual pass in `/analyticstool`.

## Code map

- `app.py`: app entry point, shared stores, Mantine provider
- `cache_config.py`: cache initialization and memoization helpers
- `pages/analyticstool.py`: primary analytics UI/callbacks
- `pages/portopt.py`: optimization UI/callbacks
- `utils/parsing.py`: file parsing and periodicity detection
- `utils/returns.py`: return conversions and compounding
- `utils/statistics.py`: metrics calculations
- `utils/optimization.py`: optimization engine and model logic

## Working rules

- Preserve callback IDs and store schemas unless migration is intentional and updated everywhere.
- Keep JSON/store payload compatibility across pages (`analyticstool` and `portopt` share state).
- Avoid broad refactors in large callback files; patch the smallest safe section.
- Add concise comments only when logic is non-obvious.
- Do not introduce new dependencies unless necessary.
- Do not mutate or delete database table data from web application runtime/callback code. Any table creation, backfill, truncate, delete, or reseed operation must live in explicit setup/migration scripts (e.g., `init_local_cma_db.py`) and never in page interaction paths.

## Data and behavior expectations

- Input files are date-indexed returns series (CSV/XLS/XLSX).
- Values may be decimals or percent-formatted; parsing should normalize safely.
- Daily data can be resampled to weekly/monthly; monthly data must not be upsampled.
- Appending data should preserve existing dataset periodicity rules.

## Performance guidance

- Prefer vectorized pandas operations.
- Reuse caching/memoization where existing code already applies it.
- Avoid expensive recomputation in callbacks when inputs are unchanged.

## UI/change safety checklist

Before finishing:
1. Confirm app starts (`python app.py`).
2. Confirm the edited workflow executes without callback errors.
3. Run relevant test scripts for touched logic.
4. Check there are no obvious regressions in tab rendering or series selection behavior.
