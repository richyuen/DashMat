# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashMat is a Python dashboard for working with market returns time series data. Built with:
- **Dash** - Plotly's framework for building analytical web applications
- **Dash Mantine Components (DMC)** - Modern UI component library
- **Dash AG Grid Enterprise** - Advanced data grid for displaying returns
- **pandas** - Data manipulation and time series handling

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

# Run with debug mode
conda run -n dashmat python app.py --debug
```

## Application Functionality

### Core Features

1. **File Upload**: Button to upload Excel (.xlsx) or CSV files containing returns data
2. **Series Selection**: MultiSelect to choose which series to include in analysis
3. **Benchmark Assignment**: Assign a benchmark to each selected series (can be any loaded series)
4. **Returns Type**: Toggle between Total Returns and Excess Returns (vs assigned benchmark)
5. **Periodicity Conversion**: Dropdown to convert daily returns to different periodicities
6. **Append Data**: Additional uploads append new series as columns to the right
7. **Export**: Download button to export displayed data as Excel

### UI Structure

- **Controls Section**: Upload, Periodicity, Returns Type, Download
- **Series Selection Section** (collapsible): MultiSelect + benchmark assignment dropdowns
- **Tabs**: Returns grid, Statistics grid

### Data Format

- **Rows**: Dates (daily or monthly)
- **Columns**: Series names (e.g., "SPY", "AGG", "GLD")
- **Values**: Returns in decimal (0.05) or percent with % sign (5%)
- Percent signs are detected and values converted to decimal internally

### Periodicity Rules

| Original Data | Allowed Conversions |
|---------------|---------------------|
| Daily | Daily, Weekly (Mon-Fri EOW options), Monthly |
| Monthly | Monthly only (no upsampling) |

**Weekly End-of-Week Options**: Monday, Tuesday, Wednesday, Thursday, Friday

### Key Logic

- **Percent Detection**: Check if any cell contains '%' character, strip and divide by 100
- **Date Parsing**: Auto-detect date format, determine if daily or monthly frequency
- **Resampling**: Use pandas `.resample()` for periodicity conversion
  - Weekly: `'W-MON'`, `'W-TUE'`, `'W-WED'`, `'W-THU'`, `'W-FRI'`
  - Monthly: `'ME'` (month end)
- **Compounding**: Convert returns to growth factors `(1 + r)`, compound with `.prod()`, subtract 1
- **Append Logic**: Outer join on date index when appending new series

### State Management

Track these in `dcc.Store`:
- `raw-data-store`: Original uploaded returns (preserves daily granularity if available)
- `original-periodicity-store`: Highest frequency available (daily or monthly)
- `benchmark-assignments-store`: Dict mapping series name to benchmark name

### Statistics Tab

Calculates per-series statistics using the assigned benchmark:

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
| Information Ratio | Annualized excess / Tracking error |
| Hit Rate | % of positive returns |
| Hit Rate (vs Benchmark) | % of periods outperforming benchmark |
| Best/Worst Period Return | Max/min single period return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Skewness/Kurtosis | Distribution shape metrics |
| 1Y/3Y/5Y metrics | Trailing period versions of key stats |

## Architecture

```
app.py                  # Entry point, layout, and callbacks
utils/
    parsing.py          # CSV/Excel parsing, percent detection, periodicity detection
    returns.py          # Return calculations, compounding, resampling
    statistics.py       # Statistics calculations (returns, risk, ratios)
```

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
