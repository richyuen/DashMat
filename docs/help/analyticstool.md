# Analytics Tool

The Analytics Tool is the core return-series analysis workspace in DashMat. Use it to load return histories, align periodicity and date ranges, compare absolute and relative streams, and review statistics, rolling behavior, drawdowns, correlations, factor relationships, and regime behavior.

## Typical workflow

1. Load data from file, database-backed imports, or existing sources.
2. Open Series Selection and choose the working series, benchmarks, long-short settings, and volatility scaling flags.
3. Set periodicity, returns mode, volatility target, and date range.
4. Validate the transformed dataset in Statistics and Returns.
5. Use the remaining tabs for deeper analysis and export the workbook when ready.

## Data requirements

- Supported files are CSV, XLS, and XLSX.
- Rows should be dates and columns should be series names.
- Values can be decimals or percent-formatted values.
- Periodicity is auto-detected on upload.
- Daily data can be converted to weekly or monthly.
- Monthly data is not upsampled to daily.
- When appending daily data to an existing monthly dataset, the daily series is resampled to monthly.

## Series Selection

- Use Series Selection to include or exclude series from the working analysis set.
- Drag and drop rows to control display order.
- Assign a benchmark to any series for relative analysis.
- Enable Long-Short to transform a series into the series-minus-benchmark stream and treat it as an absolute return stream.
- Enable per-series volatility scaling if that series should use the global Vol Scaler target.

## Core controls

### Periodicity

Periodicity controls the frequency used across the pages. The available options depend on the loaded dataset and any conversions already applied.

### Returns mode

- Total Returns uses each selected series directly.
- Excess Returns uses the arithmetic difference between a series and its benchmark for eligible pages.
- Long-Short uses the configured long-short stream as the working absolute return series.

### Vol Scaler

Set Vol Scaler to a non-zero percentage to scale enabled series to a target annualized volatility. Set it to zero to disable scaling.

### Date range shortcuts

- Common Range limits the analysis to dates where all selected series overlap.
- Common Daily jumps to the overlap where all selected series are in daily phase and sets periodicity to Daily (Trading).
- Max Range uses the earliest start and latest end available across selected series.

## Statistics page

Statistics are calculated on the currently selected periodicity and date range. If a benchmark is assigned, series and benchmark are aligned to overlapping dates before any relative metric is shown.

### Key metrics

- Cumulative Return is the compounded total return over the selected sample.
- Annualized Return uses calendar-day annualization for daily and weekly data and period-based annualization for monthly data.
- Annualized Volatility is the sample standard deviation multiplied by the square root of periods per year.
- Sharpe Ratio uses the session risk-free proxy BCTBill13.
- Sortino Ratio uses downside deviation based on periods below zero.
- Relative metrics such as Excess Return, Tracking Error, Information Ratio, Correlation, and Hit Rate vs Benchmark are blank when no valid benchmark exists.
- Maximum Drawdown is computed from cumulative wealth relative to its running peak.

### Trailing windows

1Y, 3Y, and 5Y metrics reuse the same base formulas over trailing windows. If history is insufficient, the value is blank.

## Returns page

The Returns page shows the transformed time-series table at the current periodicity. Values reflect the active returns mode, date filters, long-short settings, and volatility scaling choices.

## Rolling page

The Rolling page supports 3M, 6M, 1Y, 3Y, 5Y, and 10Y windows across metrics such as total return, excess return, volatility, tracking error, Sharpe Ratio, Sortino Ratio, Information Ratio, and Correlation. Relative metrics require a valid benchmark.

## Calendar Year page

- Annual view compounds returns within each year.
- Partial years are removed.
- Daily data requires near-full first and last years.
- Monthly view shows one selected series with Jan through Dec columns by year.

## Growth of $1 page

Growth of $1 builds cumulative wealth as the running product of `1 + r_t`. The chart shows the compounded path of each selected series and, where applicable, benchmark comparison views.

## Drawdown page

Drawdown is computed as current wealth divided by running peak minus one. Total mode uses each series directly. Excess mode with a valid benchmark uses geometric relative wealth before drawdown is calculated.

## Correlation page

- Heatmap view supports correlation and covariance matrices.
- Exponential weighting is available for matrix estimation.
- Scatter Matrix view shows pairwise scatter plots, diagonal histograms, and correlation labels.

## Factor Analysis page

- Any imported series can be used as the factor, including non-selected series.
- Edit factors supports session and database-backed factor definitions from SEC_FACTOR components.
- Box Plot mode buckets factor observations into quantiles and shows Tukey box plots.
- Scatter mode plots factor values against each selected series with an OLS trend line.
- Factor Transform supports raw values or global z-score normalization.

## Regime Analysis page

- Edit regimes supports session and database-backed regime definitions.
- HMM on PC1 fits a hidden Markov model on the first principal component of a selected universe.
- Quantile on PC1 and Quantile on Single Series split observations into ordered state buckets.
- Outputs include settings, per-regime statistics, timeline, transition probabilities, and run-duration summary.

## Export

Use File > Download Excel to export the current analytics outputs to a multi-sheet workbook. The workbook includes tabular outputs for statistics, returns, rolling views, calendar year, growth, drawdown, correlation and covariance, factor analysis, and regime analysis.
