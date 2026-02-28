# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashMat is a Python dashboard for working with market returns time series data. It now has a shared landing/import page plus three workspace pages:
- `/dashmat` - shared landing page and workspace handoff
- `/analyticstool` - returns analytics
- `/portopt` - portfolio optimization
- `/regression` - regression analysis

Built with:
- **Dash 2.14+** - Plotly's framework for building analytical web applications
- **Dash Mantine Components (DMC)** - Modern UI component library
- **Dash AG Grid 31+** - Advanced data grid for displaying returns
- **Dash Iconify** - Icon library
- **pandas 2.0+** - Data manipulation and time series handling
- **riskfolio-lib 6.0+** - Portfolio optimization (risk parity, HRP, CVaR, etc.)
- **scipy** - Scientific computing for distribution metrics
- **Flask-Caching** - Performance optimization via memoization
- **pandas_market_calendars** - NYSE trading calendar for daily_trading periodicity
- **openpyxl, xlsxwriter** - Excel file handling
- **SQLAlchemy** - Database access for raw CMA/portfolio imports

## Environment Setup

```bash
# Create conda environment
conda create -n dashmat python=3.11 -y
conda activate dashmat

# Install dependencies
pip install -r requirements.txt
```

## Common Commands

```bash
# Run the application
conda run -n dashmat python app.py

# Run with debug mode (hot reload)
conda run -n dashmat python app.py --debug

# Install dev/test dependencies
conda run -n dashmat python -m pip install -r requirements-dev.txt

# Run automated tests
conda run -n dashmat python -m pytest -q tests

# Generate test data
conda run -n dashmat python tools/data/generate_test_data.py

# Initialize local CMA/MRD test databases
conda run -n dashmat python tools/db/init_local_cma_db.py
```

## Architecture

```
DashMat/
├── app.py                    # Entry point, MantineProvider, AppShell, cache init
├── cache_config.py           # Flask-Caching setup with lazy memoize decorator
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Dev/test dependencies (pytest, pytest-cov)
├── CLAUDE.md                 # This file
├── TEST_PLAN.md              # High-level manual validation plan
├── tools/
│   ├── benchmark_callback_latency.py   # Latency benchmark harness
│   ├── data/
│   │   └── generate_test_data.py       # Benchmark return sample data downloader/exporter
│   └── db/
│       └── init_local_cma_db.py        # Local CMA/MRD database initializer
├── tests/                    # Automated test suite (unit, callbacks, scripts)
│   ├── README.md             # Test scope, commands, and coverage rules
│   ├── conftest.py
│   ├── callbacks/            # Callback-level integration tests
│   │   ├── test_analyticstool_callbacks.py
│   │   ├── test_app_shell_callbacks.py
│   │   ├── test_home_callbacks.py
│   │   ├── test_portopt_callbacks.py
│   │   ├── test_restricted_page.py
│   │   ├── test_sheet_import_helpers.py
│   │   └── test_upload_smoke_callbacks.py
│   ├── scripts/              # Integration / smoke tests
│   └── unit/                 # Pure unit tests
├── pages/
│   ├── __init__.py
│   ├── home.py               # Home/portal page
│   ├── dashmat.py            # Shared landing/import page
│   ├── restricted.py         # Access-restricted placeholder page
│   ├── analyticstool.py      # Analytics workspace
│   ├── portopt.py            # Portfolio optimization workspace
│   └── regression.py         # Regression workspace
└── utils/
    ├── __init__.py
    ├── add_series_flow.py     # Add-series validation helpers
    ├── charting.py            # Plotly chart theming (apply_chart_theme)
    ├── constants.py           # Window/day mappings, index/peer constants
    ├── core_categories.py     # CoreCategories & CMA returns helpers
    ├── dashmat_welcome_modal.py  # Shared landing welcome builders and modal JS helpers
    ├── date_range_flow.py     # Date range candidate computation
    ├── excel_export.py        # Excel date formatting helpers
    ├── exponential_weighting.py  # EWM parameter normalization
    ├── optimization.py        # Portfolio optimization engine (riskfolio-lib)
    ├── parsing.py             # File parsing, percent detection, periodicity
    ├── perf_timing.py         # Performance timing utilities
    ├── page_paths.py          # Canonical route/path helpers
    ├── portfolio_series.py    # CMA portfolio-series import helpers
    ├── raw_data_imports.py    # Raw DB import workflows (factor/funds/performance)
    ├── returns.py             # Return calculations, resampling, compounding
    ├── route_intent.py        # Landing-to-workspace handoff payloads
    ├── sample_data.py         # Sample data generation for downloads
    ├── serialization.py       # Serialization and cache-key normalization
    ├── shared_metrics.py      # Shared metric definitions
    ├── statistics.py          # Statistics calculations (40+ metrics)
    └── upload_flow.py         # Workbook sheet import and merge logic
```

## Application Functionality

### Core Features

1. **File Upload**: Upload Excel (.xlsx, .xls) or CSV files containing returns data
2. **Series Selection Modal**: Select series, rename, assign benchmarks, configure long-short
3. **Benchmark Assignment**: Assign any loaded series as a benchmark
4. **Returns Type**: Toggle between Total Returns and Excess Returns (vs benchmark)
5. **Periodicity Conversion**: Convert daily returns to weekly (Mon-Fri options) or monthly
6. **Long-Short Analysis**: Calculate difference between series and benchmark
7. **Volatility Scaling**: Scale returns to target volatility percentage
8. **Append Data**: Additional uploads append new series; daily data auto-resamples to monthly when appending to monthly data
9. **Excel Export**: Multi-sheet export with all tabs
10. **Sample Data Download**: File menu offers sample daily or monthly data files (Daily1-6 or Monthly1-6)
11. **Save/Load Session**: File menu can dump/restore all sessionStorage to a JSON blob
12. **Role-based Access**: `userinfo` store controls routing; `Test` role is redirected to `/restricted`

### Portfolio Import Source Notes

- `Add portfolios (peer)` reads `PeerTS`.
- `Add portfolios (index)` reads `IndexTS`.
- `Add portfolios (other)` is source-dispatch based by `Portfolios.PortfolioVintage`; currently implemented source is `AltTS`.
- For `other` with `PortfolioVintage='AltTS'`:
  - Portfolio series comes from `AltTS` with `Item='PortRet'`.
  - Benchmark series (if included) comes from `AltTS` with the same portfolio key and `Item='BenchRet'`.
  - Benchmark naming follows `<Portfolio>_BM`.

### UI Structure

**Global AppShell Header** (defined in `app.py`):
- Left: "DashMat" title text
- Right: `dmc.ColorSchemeToggle` (light/dark) + navigation `dmc.Menu` (Home, Analytics Tool, Portfolio Optimization)

**Per-Page Menu Bar** (inside each page):
- File: Add series, Download sample data (daily/monthly), Download Excel, Save Session, Load Session, Exit
- Edit: Clear all series, Clear storage
- View: (page-specific controls)
- Help: User Guide modal

**Main Controls (Accordion)**:
- Series Selection button → Opens modal
- Periodicity dropdown (auto-populated based on data)
- Returns Type toggle: Total/Excess
- Vol Scaler input: 0-100%
- Date range pickers

**Tabs**:
1. **Statistics** - 40+ metrics per series in AG Grid
2. **Returns** - Time series returns data grid
3. **Rolling** - Rolling metrics with configurable window/metric, chart/table toggle
4. **Calendar Year** - Annual or monthly heatmap view (Jan-Dec columns)
5. **Growth of $1** - Compound growth chart/table
6. **Drawdown** - Drawdown series chart/table
7. **Correlation** - Correlation heatmap or correlogram (scatter matrix)

### Series Selection Modal

- Checkboxes to select series with move-up/down ordering
- For each selected series:
  - Rename input field
  - Benchmark dropdown (any series or "None")
  - Long-Short toggle
  - Vol Scaling toggle
- OK/Cancel applies or discards changes

### Data Format

- **Rows**: Dates (daily or monthly)
- **Columns**: Series names (e.g., "SPY", "AGG", "GLD")
- **Values**: Returns in decimal (0.05) or percent with % sign (5%)
- Percent signs are auto-detected and converted to decimal internally

### Periodicity Rules

| Original Data | Allowed Conversions |
|---------------|---------------------|
| Daily | Daily, Daily Trading, Weekly (Mon-Fri EOW options), Monthly |
| Monthly | Monthly only (no upsampling) |

**Weekly End-of-Week Options**: Monday, Tuesday, Wednesday, Thursday, Friday

**Auto-resampling**: When appending daily data to an existing monthly dataset, daily data is automatically resampled to monthly frequency.

### Portfolio Optimization (pages/portopt.py)

Separate page (`/portopt`) for running portfolio optimizations on loaded series.

**Optimization Controls** (three-row layout):
- Row 1: Portfolio Name, Model, Exp Wt Cov, Half-Life (Periods)
- Row 2: Window (Expanding/Rolling/Full), Fill In-Sample, Window Size (Periods), Opt Step + Unit, Missing Data
- Row 3: Run button

**Opt Step Unit**: Dropdown next to Opt Step input with two modes:
- **Months** (default): Rebalance points snap to calendar month-end dates (`pd.offsets.MonthEnd`)
- **Periods**: Raw period count stepping (legacy behavior)

**Models**: Risk Parity, Factor Risk Parity, Hierarchical Risk Parity, Maximize Sharpe Ratio, Minimize CVaR, Equal Weight

**Window Types**:
- **Rolling**: Fixed-size estimation window slides forward
- **Expanding**: Estimation window grows from start
- **Full**: Single window using all data

**Weight Constraints**: Per-series min/max weight bounds and force-to-max toggle, configured in Series Selection modal

**Results Storage** (`po-results-store`):
```python
{
    "PortfolioName": {
        "window_weights": [{"apply_start": "...", "apply_end": "...", "weights": {...}}, ...],
        "returns_json": "<JSON Series>",
        "config": {...},
    }
}
```

**Cross-page sync**: `po_sync_results_with_raw_data` callback fires on page load (`po-page-load-trigger`) to prune portfolios deleted from other pages.

## State Management

### Global Stores (defined in `app.py`, `dashmat-` prefix)

```python
dashmat-raw-data-store             # Original uploaded data (JSON string), session storage
dashmat-original-periodicity-store # Auto-detected: 'daily' or 'monthly', session storage
dashmat-raw-data-summary-store     # Lightweight client summary for date-range init, memory storage
dashmat-pending-new-series-store   # Queued series to add, session storage
dashmat-saved-series-cache-store   # Cached series data, session storage
dashmat-route-intent-store         # Landing/workspace handoff payload, session storage
userinfo                           # Role-based access info (e.g., {"role": "Admin"})
```

### Analytics Tool Stores (defined in `analyticstool.py`, `at-` prefix)

```python
# Core series state
at-series-select-value-store       # List of selected series names
at-series-order-store              # Display ordering of series
at-benchmark-assignments-store     # Dict: {series_name: benchmark_name}
at-long-short-store                # Dict: {series_name: is_long_short}
at-vol-scaling-assignments-store   # Dict: {series_name: apply_vol_scaling}

# UI controls
at-periodicity-value-store         # Current periodicity selection
at-periodicity-load-sync-dummy     # Sync trigger for periodicity on load
at-returns-type-value-store        # 'total' or 'excess'
at-vol-scaler-value-store          # Volatility scaling percentage
at-date-range-store                # Start/end date filters

# Tab states
at-active-tab-store                # Current active tab
at-rolling-window-store            # Selected rolling window
at-rolling-metric-store            # Selected rolling metric
at-rolling-return-type-store       # 'annualized' or other
at-rolling-chart-switch-store      # Chart vs table view
at-drawdown-chart-switch-store     # Chart vs table view
at-growth-chart-switch-store       # Chart vs table view
at-monthly-view-store              # Annual vs monthly calendar view
at-monthly-series-store            # Selected series for monthly view

# Page lifecycle
at-state-ready-store               # True once page state is initialized
at-statistics-loaded-store         # True once statistics tab has rendered
at-download-enabled-store          # Controls Excel download availability
at-first-load-store                # True on first page load

# Modal draft state (temp stores hold changes until OK is clicked)
at-temp-series-select
at-temp-benchmark-assignments-store
at-temp-long-short-store
at-temp-vol-scaling-assignments-store
at-temp-series-order-store
at-temp-deleted-series-store
at-portfolio-add-mode-store        # Tracks which add-portfolio source is active
```

### Portfolio Optimization Stores (`po-` prefix)

```python
po-series-select, po-series-order-store
po-benchmark-assignments-store, po-cmabench-assignments-store
po-long-short-store, po-vol-scaling-assignments-store
po-min-wt-store, po-max-wt-store, po-force-max-store
po-results-store                   # Optimization results keyed by portfolio name
# ... temp stores for modal draft state
```

### State Management Pattern

1. **Separation**: Persistent stores (session storage) vs transient UI component state
2. **Sync callbacks**: Map store → UI and UI → store bidirectionally
3. **Modal pattern**: Temp stores hold draft changes until user confirms with OK
4. **JSON serialization**: DataFrames stored as JSON for browser transport
5. **Page load trigger**: `dcc.Interval` (e.g., `po-page-load-trigger`) fires once on navigation to sync cross-page state

## Key Logic

### Parsing (utils/parsing.py)

- **Percent Detection**: Check if any cell contains '%', strip and divide by 100
- **Date Parsing**: Auto-detect format, set as DatetimeIndex
- **Periodicity Detection**: Sample first 5 rows, median diff > 20 days = monthly

```python
parse_uploaded_file(contents, filename)  # Entry point
convert_percents_to_decimals(df)         # Format handling
detect_periodicity(df) -> str            # Returns 'daily'/'monthly'
```

### Upload Flow (utils/upload_flow.py)

```python
import_selected_workbook_sheets(contents, filename, selected_sheets)  # Multi-sheet import
import_single_upload(contents, filename)                               # Single-file import
merge_uploaded_with_existing(existing_data, existing_periodicity, new_df)  # Merge result
```

### Date Range Flow (utils/date_range_flow.py)

```python
compute_date_range_candidates(raw_data, periodicity, selected_series) -> dict
resolve_initial_range(candidates, stored_range) -> (start, end)
resolve_button_range(candidates, button_id) -> (start, end, changed)
```

### Constants (utils/constants.py)

Window mappings extracted for use across rolling calculations:

```python
WINDOW_MAP_DAYS    # {"3m": "91D", "6m": "183D", ...} for pd.rolling
WINDOW_DAYS_MAP    # {"3m": 91, "6m": 183, ...} for min_periods
WINDOW_YEARS_MAP   # {"3m": 0.25, "6m": 0.5, ...} for annualization
```

### Returns Calculations (utils/returns.py)

- **Compounding**: `(1 + r).prod() - 1` using numpy for performance
- **Resampling**: pandas `.resample()` with partial period masking
  - Weekly: `'W-MON'`, `'W-TUE'`, `'W-WED'`, `'W-THU'`, `'W-FRI'`
  - Monthly: `'ME'` (month end)
- **Excess Returns**: Arithmetic difference (series - benchmark)
- **Long-Short**: Treats series-benchmark difference as absolute stream
- **Volatility Scaling**: Scale returns to achieve target annualized vol

```python
get_working_returns()              # Core calculation engine
calculate_excess_returns()         # Excess returns calculation
calculate_rolling_returns()        # Rolling stats (3m-10y windows)
calculate_calendar_year_returns()  # Annual returns by year
create_monthly_view()              # Jan-Dec monthly heatmap
df_to_json() / json_to_df()       # DataFrame serialization helpers
```

**Annualization Factors**:
- Daily: 252
- Weekly: 52
- Monthly: 12

### Serialization (utils/serialization.py)

Centralizes stable serialization for cache keys and callback payloads. Converts objects to deterministic JSON-serializable structures with sorted keys.

### Exponential Weighting (utils/exponential_weighting.py)

```python
normalize_decay_input(value) -> float  # Clamp/normalize decay input
decay_input_mode(value) -> str         # "halflife" or "decay" display mode
resolve_ewm_params(value) -> dict      # Returns {"halflife": ...} or {"alpha": ...}
```

### Chart Theming (utils/charting.py)

```python
apply_chart_theme(fig, theme)  # Applies "plotly_dark"/"plotly_white" template + transparent bg
```

All server-side chart callbacks take `State("mantine-provider", "forceColorScheme")` to read the current theme.

### Excel Export (utils/excel_export.py)

```python
format_mdy_date(value)  # Format date-like values as m/d/yyyy for Excel consistency
```

### Sample Data (utils/sample_data.py)

```python
generate_sample_returns(periodicity)  # Create daily or monthly sample data (6 series)
create_sample_excel(periodicity)      # Package as downloadable Excel bytes
```

Series are named `Daily1-6` or `Monthly1-6` depending on periodicity.

### Statistics (utils/statistics.py)

| Statistic | Description |
|-----------|-------------|
| Start/End Date | Data range |
| Number of Periods | Count of return observations |
| Cumulative Return | Total compounded return |
| Annualized Return | Geometric annualized return |
| Annualized Excess Return | Annualized return of excess series |
| Annualized Volatility | Std dev × √periods_per_year |
| Annualized Tracking Error | Std dev of excess returns × √periods_per_year |
| Sharpe Ratio | Annualized return / Annualized volatility (rf=0) |
| Sortino Ratio | Annualized return / Downside deviation |
| Information Ratio | Annualized excess / Tracking error |
| Correlation | Pearson correlation vs benchmark |
| Hit Rate | % of positive returns |
| Hit Rate (vs Benchmark) | % of periods outperforming benchmark |
| Best/Worst Period Return | Max/min single period return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Skewness/Kurtosis | Distribution shape metrics (scipy.stats) |
| 1Y/3Y/5Y metrics | Trailing period versions of key stats |

## Dark Mode / Theming

- `dmc.ColorSchemeToggle` is rendered in the **global AppShell header** (`app.py`), not per-page
- The `dmc.MantineProvider` has `id="mantine-provider"`; its `forceColorScheme` is driven by the toggle
- Chart callbacks read `State("mantine-provider", "forceColorScheme")` for theme-aware rendering
- `apply_chart_theme(fig, theme)` in `utils/charting.py` handles Plotly template switching
- AG Grid dark mode via `assets/style.css` using `[data-mantine-color-scheme="dark"]` selector
- `dmc.pre_render_color_scheme()` is called at module level in `app.py` to prevent flash

## Save/Load Session

- Both pages have File > Save Session / Load Session
- **Save**: clientside callback dumps all sessionStorage keys to a JSON blob download
- **Load**: hidden `dcc.Upload` + clientside callbacks to restore sessionStorage and reload

## Access Control

- `userinfo` store (in `app.py`) holds `{"role": "Admin"}` (or `"Test"`)
- `Test` role is redirected to `/restricted?target=<page>` via `guard_protected_pages` callback
- Navigation menu links update dynamically via `update_global_nav_links` callback

## Caching Strategy (cache_config.py)

```python
# Initialize cache
init_cache(app.server)  # SimpleCache, 300s timeout, 500 items max

# Lazy memoize decorator for module-level use (works before cache init)
@cache_config.memoize(timeout=300)
def expensive_calculation(...):
    ...
```

The memoize decorator uses MD5-hashed cache keys from function arguments and defers cache access until call time, allowing decoration at import time before the cache is initialized.

## AG Grid Configuration

```python
dag.AgGrid(
    id="returns-grid",
    columnDefs=[{"field": "Date", "pinned": "left"}] + series_columns,
    rowData=returns_df.to_dict("records"),
    defaultColDef={
        "sortable": True,
        "resizable": True,
    },
    dashGridOptions={
        "animateRows": True,
        "pagination": True,
        "paginationPageSize": 100,
    }
)
```

## Code Conventions

### Callbacks

- Clientside callbacks handle instant UI toggles (store ↔ component sync) for performance
- Server-side callbacks handle data processing, chart rendering, and complex logic
- Use `prevent_initial_call=True` where appropriate
- Callbacks return `dash.no_update` to skip unnecessary updates
- `allow_duplicate=True` required for secondary outputs targeting the same component

### Cross-Page Callback Pitfalls

- **Never** output to a store that another callback reads AND writes (circular dependency → 500 errors)
- **Never** mix Output targets from main layout and page layout in same callback
- **Never** reference components from other pages as Outputs — use State-only for cross-layout reads
- After changing callback outputs, clear `__pycache__`, `./cache`, AND browser site data — Dash caches callback graphs aggressively
- Page-specific callbacks don't fire when that page isn't rendered — use a one-shot `dcc.Interval` to sync on navigation

### Recent Learnings

- `/dashmat` is the only visible welcome/import page. The workspace pages no longer show local welcome screens; when raw data is missing they redirect to `/dashmat?module=<tool>`.
- Landing-to-workspace handoff uses `dashmat-route-intent-store` plus per-page `*-route-intent-consumed-token-store`. Keep route intent writes and workspace navigation in the same callback response to avoid races.
- Landing-origin non-file modal flows should return to `/dashmat?module=<tool>` if the modal closes without creating raw data. The shared empty-workspace router handles this using the consumed route-intent token.
- `File -> New Session` should clear DashMat-owned session keys and navigate directly to `landing_href(<module>)` with `window.location.replace(...)`. Preserve `userinfo` and unrelated session keys. `bctbill13-cache-store` is legacy-only cleanup; the live store is `dashmat-saved-series-cache-store`.
- Analytics import semantics now differ from the older staged-import model:
  - successful import callbacks commit `dashmat-raw-data-store` immediately
  - working analytics state is staged in `at-pending-working-config-store`
  - Series Selection `OK` applies the pending working config
  - Series Selection `Cancel` keeps raw data but leaves working state unchanged
- Shared module-switch date-range init uses `dashmat-raw-data-summary-store` plus hash-keyed metadata in `utils/date_range_flow.py`. The summary store is intentionally `memory`, not `session`.
- For cross-page SPA navigation, Dash Pages routing is driven by `_pages_location`. Updating a page-local `dcc.Location` changes the URL but does not switch the rendered page content.
- Global app navigation now uses `dcc.Link(refresh=False)` wrappers with `global-navbar-pretrade-*` IDs. Keep restricted-role href handling in `update_global_nav_links(...)` and `guard_protected_pages(...)`.
- When seeding `dcc.Store` values in browser automation, write the same format Dash writes to `sessionStorage`: `JSON.stringify(value)`. Writing raw strings directly can make stores hydrate with the wrong type and cause misleading callback errors.
- Dash callback outputs must target components that already exist in the layout. A tab lazy-mount approach that leaves output-target IDs absent from the layout will fail at runtime even if Python tests pass. To safely defer heavy content, keep stable output targets mounted and lazy-build behind host `children` instead of removing the target components themselves.

### Data Flow

1. File upload → `dashmat-raw-data-store`
2. UI controls (periodicity, returns-type) → `at-*` stores (via clientside callbacks)
3. Stores → calculation functions (memoized)
4. Results → grid/chart rendering

### Financial Calculations

- All returns calculations use vectorized numpy/pandas operations
- Benchmark alignment: Force common dates before relative metrics
- NaN handling: Careful `dropna()` with `min_periods` constraints
- Outer joins preserve all dates when merging series

### riskfolio-lib Notes

- `rp_optimization()` does NOT use `lowerlng`/`upperlng` for box constraints — must convert to linear inequality constraints via `ainequality`/`binequality` (format: `ainequality * w <= binequality`)
- `optimization()` (Classic) uses `lowerlng`/`upperlng` — must be numpy arrays (not DataFrames) for cvxpy compatibility in Sharpe formulation
- HRP uses `rp.HCPortfolio(returns=data).optimization(model="HRP")` — not `rp.Portfolio.optimization(model='HRP')` which silently fails
- Factor RP: call `factors_stats()` after setting `port.factors`, use `hist=False` in `rp_optimization(model="FM")`
- Always call `port.assets_stats()` before optimization
- Suppress cvxpy deprecation warnings: `warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")`

### Error Handling

- Graceful handling of missing benchmarks
- Validation before invalid operations (e.g., upsampling monthly data)
- Modal validation prevents invalid benchmark assignments

### Performance Optimizations

- Clientside callbacks for instant UI toggles (no server round-trip)
- `json.loads()` for JSON deserialization (not `eval()`)
- Cache threshold of 500 items for better hit rate
- Memoization with MD5 hash keys on expensive calculations
- Vectorized numpy operations for compounding

## Testing

```bash
# Install dev/test dependencies
conda run -n dashmat python -m pip install -r requirements-dev.txt

# Run full automated suite
conda run -n dashmat python -m pytest -q tests

# Optional: run callback latency benchmark harness
conda run -n dashmat python tools/benchmark_callback_latency.py
```

Automated test structure and coverage gate details are maintained in `tests/README.md`.
Manual workflow validation checklist remains in `TEST_PLAN.md`.

## Excel Export

Multi-sheet workbook with:
- Statistics sheet
- Returns sheet
- Rolling (configurable metric/window)
- Calendar Year
- Growth of $1
- Drawdown
- Correlation matrix
