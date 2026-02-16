# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashMat is a Python dashboard for working with market returns time series data. Built with:
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
├── app.py                    # Entry point, MantineProvider, cache init
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
│   └── README.md             # Test scope, commands, and coverage rules
├── pages/
│   ├── __init__.py
│   ├── home.py               # Welcome/portal page (links to /dashboard)
│   ├── analyticstool.py      # Main analytics dashboard (~3,700 lines)
│   └── portopt.py            # Portfolio optimization page
└── utils/
    ├── __init__.py
    ├── constants.py           # Window/day mappings for rolling calculations
    ├── optimization.py        # Portfolio optimization engine (riskfolio-lib)
    ├── parsing.py             # File parsing, percent detection, periodicity
    ├── returns.py             # Return calculations, resampling, compounding
    ├── sample_data.py         # Sample data generation for downloads
    └── statistics.py          # Statistics calculations (40+ metrics)
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

### Portfolio Import Source Notes

- `Add portfolios (peer)` reads `PeerTS`.
- `Add portfolios (index)` reads `IndexTS`.
- `Add portfolios (other)` is source-dispatch based by `Portfolios.PortfolioVintage`; currently implemented source is `AltTS`.
- For `other` with `PortfolioVintage='AltTS'`:
  - Portfolio series comes from `AltTS` with `Item='PortRet'`.
  - Benchmark series (if included) comes from `AltTS` with the same portfolio key and `Item='BenchRet'`.
  - Benchmark naming follows `<Portfolio>_BM`.

### UI Structure

**Welcome Screen**: Shown when no data loaded - icon + "Add series from file" button

**Menu Bar**:
- File: Add series, Download sample data (daily/monthly), Download Excel, Exit
- Edit: Clear all series, Clear storage
- Help: (placeholder)

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
| Daily | Daily, Weekly (Mon-Fri EOW options), Monthly |
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

### Core Data Stores (dcc.Store)

```python
# Core data
raw-data-store                    # Original uploaded data (JSON string)
original-periodicity-store        # Auto-detected: 'daily' or 'monthly'
series-select                     # List of selected series names
benchmark-assignments-store       # Dict: {series_name: benchmark_name}
long-short-store                  # Dict: {series_name: is_long_short}

# UI controls
periodicity-value-store           # Current periodicity selection
returns-type-value-store          # 'total' or 'excess'
vol-scaler-value-store            # Volatility scaling percentage
date-range-store                  # Start/end date filters
vol-scaling-assignments-store     # Dict: {series_name: apply_vol_scaling}

# Tab states
active-tab-store                  # Current active tab
rolling-metric-store              # Selected rolling metric
rolling-window-store              # Selected rolling window
drawdown-chart-switch-store       # Chart vs table view
growth-chart-switch-store         # Chart vs table view
monthly-view-store                # Annual vs monthly calendar view
monthly-series-store              # Selected series for monthly view

# Modal state
temp-series-select                # Draft selection before OK
temp-benchmark-assignments-store  # Draft assignments before OK
series-edit-mode                  # Modal open state
```

### State Management Pattern

1. **Separation**: Persistent stores vs transient UI component state
2. **Sync callbacks**: Map store → UI and UI → store bidirectionally
3. **Modal pattern**: Temp stores hold draft changes until user confirms
4. **JSON serialization**: DataFrames stored as JSON for browser transport

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
```

**Annualization Factors**:
- Daily: 252
- Weekly: 52
- Monthly: 12

### Sample Data (utils/sample_data.py)

Generates synthetic returns data for download from the File menu:

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

## Caching Strategy (cache_config.py)

```python
# Initialize cache
init_cache(app.server)  # SimpleCache, 300s timeout, 500 items max

# Lazy memoize decorator for module-level use (works before cache init)
@cache_config.memoize(timeout=300)
def expensive_calculation(...):
    ...

# CacheProxy for cache-agnostic code
cache_config.cache.memoize(timeout=300)
```

The memoize decorator uses MD5-hashed cache keys from function arguments and defers cache access until call time, allowing decoration at import time before the cache is initialized.

**Purpose**: Prevent recalculation of expensive financial operations (compounding, statistics).

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

- 59 callbacks in analyticstool.py: 34 server-side `@callback` + 25 `clientside_callback`
- Clientside callbacks handle instant UI toggles (store ↔ component sync) for performance
- Server-side callbacks handle data processing, chart rendering, and complex logic
- Use `prevent_initial_call=True` where appropriate
- Callbacks return `dash.no_update` to skip unnecessary updates

### Data Flow

1. File upload → `raw-data-store`
2. UI controls (periodicity, returns-type) → stores (via clientside callbacks)
3. Stores → calculation functions (memoized)
4. Results → grid/chart rendering

### Financial Calculations

- All returns calculations use vectorized numpy/pandas operations
- Benchmark alignment: Force common dates before relative metrics
- NaN handling: Careful `dropna()` with `min_periods` constraints
- Outer joins preserve all dates when merging series

### Error Handling

- Graceful handling of missing benchmarks
- Validation before invalid operations (e.g., upsampling monthly data)
- Modal validation prevents invalid benchmark assignments

### Performance Optimizations

- 25 clientside callbacks for instant UI toggles (no server round-trip)
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
