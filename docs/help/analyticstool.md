# Analytics Tool

The Analytics Tool is the main return-series analysis workspace in DashMat. It is designed for the common workflow of loading one or more return streams, normalizing them into a consistent analysis sample, selecting how each series should be interpreted, and then reviewing performance, risk, correlation, factor, and regime outputs from the same transformed dataset.

This page is usually the best starting point when you want to answer questions such as:

- How did a strategy perform over a specific sample?
- How does that performance change if returns are viewed as absolute, excess, or long-short?
- Which series are highly correlated or diversifying?
- How do results vary through time, by year, by factor bucket, or by market regime?

## Typical workflow

1. Import one or more return series from file or another supported source.
2. Open **Series Selection** and decide which series belong in the active analysis set.
3. Assign benchmarks where relative analysis is needed.
4. Configure optional long-short and volatility-scaling behavior per series.
5. Set **Periodicity**, **Returns Type**, **Vol Scaler**, and the date range.
6. Check **Statistics** and **Returns** first to confirm that the transformed dataset matches your expectations.
7. Move through the downstream tabs for rolling behavior, annual results, drawdowns, correlation, factor analysis, and regime analysis.
8. Export the workbook when the outputs are in the desired state.

If the downstream tabs look wrong, the issue is usually earlier in the workflow: selected series, assigned benchmarks, date overlap, returns mode, or periodicity.

## Data requirements and import behavior

### Supported input structure

- Supported file types are `CSV`, `XLS`, and `XLSX`.
- Rows should represent dates.
- Columns should represent series names.
- Values may be decimals such as `0.01` or percent-style values such as `1%`.
- Duplicate names are usually a problem because later controls and outputs identify series by name.

### Frequency rules

- Periodicity is auto-detected on upload.
- Daily data can be converted to weekly or monthly.
- Monthly data is not upsampled to daily.
- If daily data is appended into an existing monthly dataset, the daily series is resampled to monthly so the combined dataset remains internally consistent.

These rules matter because many downstream calculations depend on the current periodicity. If the frequency is not what you expected, validate the original source first and then check whether the current session already contains data at another periodicity.

### Practical import guidance

- Keep one return series per column.
- Prefer clean, unique column names before upload.
- If a workbook contains multiple sheets, confirm that the imported sheet is the intended one.
- If your values were typed as percentages in Excel, DashMat will generally normalize them correctly, but it is still worth checking the first few rows in **Returns** after import.

## Series Selection

The **Series Selection** modal is where the active analysis universe is defined. Most unexpected outputs can be traced back to this modal.

### What you can control

- Include or exclude series from the working analysis set.
- Drag and drop rows to control display order.
- Assign a benchmark to any series.
- Enable **Long-Short** so that a series is transformed to `series - benchmark` and then treated as an absolute stream.
- Enable **Scale Vol** for series that should use the global **Vol Scaler** target.

### Benchmark assignment

Benchmarks are required for most relative analysis. If a series does not have a valid benchmark, relative fields such as excess return, tracking error, information ratio, or benchmark hit rate will remain blank where those metrics depend on benchmark alignment.

Benchmark assignment is also what enables two related but distinct workflows:

- **Excess Returns**: arithmetic difference between the series and the benchmark.
- **Long-Short**: the benchmark-relative stream is treated as the new absolute return series for downstream analysis.

Use excess returns when you want to measure relative performance while still thinking of the original series as the main object of study. Use long-short when the spread itself is the strategy.

### Volatility scaling

Vol scaling is controlled in two places:

- The page-level **Vol Scaler** target.
- The per-series **Scale Vol** flag in Series Selection.

A series is only scaled when both are active. This is useful when you want to compare different strategies on a common risk budget without forcing every series into the same target.

## Core controls

The controls at the top of the page define the transformed dataset used by almost every tab.

### Periodicity

**Periodicity** controls the working frequency used across the page. Available options depend on the imported data and the conversions that are allowed from that source frequency.

Use this control carefully:

- Daily to weekly or monthly changes the return stream itself, not just the display.
- Many metrics, especially annualized statistics and rolling windows, will change when periodicity changes.
- If a comparison is meant to be stable across tabs, change periodicity first and then review all tabs under that same setting.

### Returns Type

The Analytics Tool supports three major return interpretations:

- **Total Returns**: each selected series is analyzed directly.
- **Excess Returns**: series minus benchmark where a valid benchmark exists.
- **Long-Short**: the configured long-short stream becomes the series analyzed throughout the page.

This control has large downstream effects. For example:

- Statistics will show different return and risk values.
- Growth of $1 and Drawdown will use a different return stream.
- Relative metrics may disappear if a benchmark is missing or invalid under the current setup.

If you are trying to reconcile values across exports or screenshots, always confirm the active returns mode first.

### Vol Scaler

Set **Vol Scaler** to a non-zero percentage to target an annualized volatility level for any series with **Scale Vol** enabled. Set it to zero to disable scaling entirely.

Vol scaling is useful when:

- comparing managers with different realized risk profiles
- testing whether apparent outperformance is mostly a leverage effect
- putting multiple strategies onto a comparable volatility footing

Vol scaling is not a free lunch. It changes the actual return path used in later calculations, so annualized return, drawdown, Sharpe Ratio, and rolling outputs can all move materially.

### Date range controls

DashMat offers both explicit date inputs and shortcut buttons.

- **Common Range** restricts the sample to dates shared by all selected series.
- **Common Daily** jumps to the overlap where all selected series are in daily phase and sets periodicity to daily.
- **Max Range** uses the earliest start and latest end available across the selected series.

These shortcuts are useful because benchmark-relative outputs only work on overlapping dates. If series start and end at different times, the statistics shown on one tab may be based on fewer rows than you intuitively expected.

## Statistics

The **Statistics** tab is the fastest way to validate the transformed sample. It should usually be the first stop after changing any top-level control.

### What the tab shows

- cumulative and annualized return measures
- volatility and downside-risk measures
- risk-adjusted ratios such as Sharpe and Sortino
- relative metrics when a benchmark exists
- trailing-window statistics where enough history is available

### How to interpret key fields

- **Cumulative Return** is the compounded return over the current sample.
- **Annualized Return** uses calendar-day annualization for daily and weekly data and period-based annualization for monthly data.
- **Annualized Volatility** is based on sample standard deviation scaled by periods per year.
- **Sharpe Ratio** uses the session risk-free proxy `BCTBill13`.
- **Sortino Ratio** focuses on downside deviation rather than total volatility.
- **Maximum Drawdown** is derived from cumulative wealth relative to its running peak.

### Relative metrics

Metrics such as excess return, tracking error, information ratio, correlation, and hit rate versus benchmark require a valid benchmark and sufficient overlapping data. If those fields are blank, look for one of these issues:

- no benchmark assigned
- benchmark assigned but not present in the working dataset
- insufficient overlap after date filtering
- returns mode or long-short setup no longer aligned with the benchmark relationship

### Trailing windows

Trailing windows such as `1Y`, `3Y`, and `5Y` reuse the same formulas on a shorter sample. If the sample is too short, the value is intentionally left blank rather than extrapolated.

## Returns

The **Returns** tab displays the transformed time-series table that feeds most of the downstream views. This is the best place to confirm:

- the active date range
- periodicity conversion
- excess-return or long-short behavior
- volatility-scaling behavior

If the numeric values in downstream charts seem suspicious, verify this table before assuming the chart is wrong.

## Rolling

The **Rolling** tab shows how metrics evolve over time using windows such as `3M`, `6M`, `1Y`, `3Y`, `5Y`, and `10Y`.

Typical uses:

- identifying whether a strategy's risk-adjusted performance is stable or episodic
- comparing rolling volatility and drawdown sensitivity across series
- evaluating whether excess return or information ratio is persistent

Relative metrics on this tab still require a valid benchmark. If you switch from total to excess mode, expect the rolling series to change substantially.

## Calendar Year

The **Calendar Year** tab is designed for annual attribution of performance by year.

- Annual view compounds returns within each calendar year.
- Partial years are removed.
- For daily data, the first and last year generally need enough coverage to count as a full year.
- Monthly mode shows one selected series with `Jan` through `Dec` columns by year.

This tab is helpful when the full-sample statistics hide sequencing effects. Two strategies can have similar long-run returns but very different year-by-year patterns.

## Growth of $1

The **Growth of $1** tab compounds the active return stream as a running product of `1 + r_t`. It answers a simple question: what path would one dollar have followed under the current settings?

Use this tab to compare:

- absolute growth across strategies
- the impact of switching from total to excess or long-short mode
- the effect of volatility scaling on the path of wealth

When returns mode changes, this chart may change in both slope and shape because the underlying series itself has changed.

## Drawdown

The **Drawdown** tab converts the active return stream into a peak-to-trough loss path.

- In total-return mode, drawdown is based on each series directly.
- In excess mode with a valid benchmark, geometric relative wealth is used before drawdown is computed.

This tab is especially useful for identifying whether attractive average returns came from a smooth process or from a path with severe interim losses.

## Correlation

The **Correlation** area supports both matrix-style and scatter-style cross-series analysis.

### Heatmap and covariance

- The matrix view can show correlation or covariance.
- Exponential weighting can be used when you want recent observations to matter more than older ones.
- These outputs are built from the currently transformed dataset, so periodicity, date range, and returns mode all matter.

### Scatter Matrix

The scatter-matrix view helps identify:

- linear relationships
- asymmetry and outliers
- clusters of similar behavior
- pairs that look more or less related than the headline correlation suggests

## Factor Analysis

The **Factor Analysis** tab is for asking how selected series behave relative to another imported series used as a factor.

### What can be used as a factor

Any imported series can be used as the factor, including series that are not part of the active selected universe.

### Main modes

- **Box Plot** buckets factor observations into quantiles and shows the distribution of selected-series returns within each bucket.
- **Scatter** plots factor values against selected-series returns and adds an OLS trend line.
- **Q-Q Plot** compares each selected series either to a fitted normal distribution or to a chosen reference series.

### Q-Q references

- **Normal** checks whether the selected series looks close to normally distributed.
- **Reference** compares the selected series against another imported or defined reference series using standardized quantiles, which is mainly a shape comparison rather than a level comparison.

### Factor transforms

Factor values can be used in raw form or globally standardized with a z-score transform. Standardization is useful when the factor's original scale is not intuitive or when you want cross-factor comparisons to be more consistent.

### Factor definitions

The **Edit factors** workflow supports both session-level and database-backed factor definitions from `SEC_FACTOR` components. If a factor-derived result looks wrong, verify both the raw underlying series and the factor-definition logic.

## Regime Analysis

The **Regime Analysis** tab groups observations into states and then summarizes behavior inside each state.

### Main regime methods

- **HMM on PC1** fits a hidden Markov model to the first principal component of a selected universe.
- **Quantile on PC1** splits the first principal component into ordered state buckets.
- **Quantile on Single Series** uses one chosen series as the regime driver.

### Typical uses

- comparing returns and risk across calm versus stressed markets
- checking whether factor relationships change across states
- measuring transition behavior and persistence of regimes

### Outputs

Regime Analysis can produce:

- settings and definition summaries
- per-regime statistics
- timeline views
- transition probabilities
- run-duration summaries

When a regime result is hard to interpret, first confirm that the chosen driver series or PC1 universe matches the economic story you are trying to test.

## Export and session workflow

Use **File > Download Excel** to export the current analytics outputs to a multi-sheet workbook. The export includes the tabular outputs for statistics, returns, rolling views, calendar year, growth, drawdown, correlation and covariance, factor analysis, and regime analysis.

A good export workflow is:

1. Set periodicity, returns mode, and date range.
2. Confirm the transformed dataset in **Returns**.
3. Confirm headline values in **Statistics**.
4. Export only after those two validation checks pass.

## Common mistakes and troubleshooting

### Relative metrics are blank

Check benchmark assignment first, then confirm there is enough overlapping history after date filtering.

### Numbers changed across tabs after a control update

That is expected when you changed periodicity, returns mode, vol scaling, or date range. Those controls change the active dataset used by all downstream tabs.

### Rolling or annual views seem incomplete

The current sample may be too short for the requested window, or partial-year rules may have removed rows intentionally.

### Growth and drawdown do not match a total-return expectation

Confirm whether the page is in total, excess, or long-short mode. Those are different return streams.

### Imported values look too large or too small

Open **Returns** and inspect the first few rows. This usually reveals whether the source file was interpreted as decimals versus percentages or whether the wrong sheet was imported.
