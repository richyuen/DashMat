# Portfolio Optimization

Portfolio Optimization is the DashMat workspace for building and reviewing portfolio allocations across historical, ex-ante, and Black-Litterman-style models. Use it to select asset series, define constraints, run optimizations, and inspect allocations, performance, attribution, risk, and frontier outputs.

## Typical workflow

1. Load return series and choose the series used as portfolio assets.
2. Open Series Selection and configure include flags, order, benchmark assignment, long-short flags, volatility scaling, and per-asset min and max weights.
3. Set periodicity, volatility target, and date range.
4. Choose the optimization model and configure windowing, missing-data handling, and optional exponential weighting.
5. Add linear constraints or ex-ante assumptions if the selected model requires them.
6. Run the optimization under a portfolio name and review the output tabs.

## Data requirements

- Input files should be date-indexed return series with one column per asset.
- Supported uploads are CSV, XLS, and XLSX.
- Workbook imports support one or more selected sheets.
- Values may be decimals or percent-style values.
- Periodicity is auto-detected.
- Daily data can be resampled to weekly or monthly.
- Monthly data is not upsampled.
- Date overlap across selected assets affects the usable sample and output availability.

## Series Selection

- Include controls whether a series participates in optimization.
- Drag and drop rows to set display order in selectors and outputs.
- Benchmark sets the comparison series used by excess-return and long-short behavior.
- Long-Short transforms a selected series to series minus benchmark.
- Scale Vol toggles per-series use of the global volatility target.
- Min Wt and Max Wt are hard optimizer bounds.
- Force Max pins an asset to its max weight and can make the problem infeasible.
- Deleting a series removes it from the working dataset and can invalidate prior results.

## Core controls

### Periodicity

Periodicity converts returns to the analysis frequency used in optimization and downstream tabs.

### Vol Scaler

Vol Scaler applies a target annualized volatility scaling percentage. Set it to zero to disable scaling.

### Date range

- Common Range uses only dates where all selected series overlap.
- Common Daily jumps to the overlap where all selected series are in daily phase and sets periodicity to Daily (Trading).
- Max Range uses the earliest start to latest end across selected series.

## Optimization controls

- Portfolio Name is the saved result key.
- Model chooses among risk-based, historical mean-variance, ex-ante, and Black-Litterman workflows.
- Exp Wt enables exponential weighting for historical estimates.
- Decay values greater than or equal to one are treated as half-life periods; smaller values are treated as lambda.
- Window options are Expanding, Rolling, and Full.
- Window Size sets the lookback used for each optimization step.
- Opt Step and Unit set rebalance frequency.
- Missing Data supports Fill NA and Fill 0 handling.

## Model guide

### Risk Parity

Balances total risk contribution across assets. It is useful when expected return forecasts are weak and diversification of risk is the main objective.

### Factor Risk Parity

Balances risk through factor structure rather than only pairwise covariance. It is useful when assets share common factor drivers and covariance estimates are noisy.

### Hierarchical Risk Parity

Clusters assets and allocates top-down for robust diversification. It can be helpful when the correlation matrix is unstable and clustering adds structure.

### Maximize Sharpe Ratio

Seeks the highest expected return per unit of volatility. It is sensitive to mean-return estimation error and outliers.

### Minimize Variance

Seeks the lowest volatility portfolio under the active constraints. It can underweight growth assets when risk control dominates return-seeking behavior.

### Minimize CVaR

Optimizes downside tail risk rather than only variance. It requires enough data for stable tail estimation.

### Equal Weight

Uses a simple `1/N` allocation baseline. It is a useful neutral benchmark when estimation uncertainty is high.

### Ex Ante Mean-Variance

Uses forward-looking expected returns and risk assumptions that you provide directly. Assumption consistency matters: unrealistic volatilities, correlations, or non-symmetric matrices can invalidate outputs.

### Black-Litterman

Combines prior assumptions with user views and confidence levels to produce posterior return and covariance inputs. Weak view specification or extreme confidence values can dominate priors unexpectedly.

## Linear constraints

Linear constraints add rows with Min and Max bounds plus one coefficient column per selected asset. Each row enforces:

`Min <= sum(coef_i * weight_i) <= Max`

Use these constraints to control grouped exposures or policy bands.

## Ex ante inputs

- Ex Ante Mean-Variance and Black-Litterman support user-provided return and risk assumptions.
- Input Mode toggles between return plus covariance and return plus volatility plus correlation.
- Use upload and estimate helpers to seed assumptions, then verify magnitudes and matrix symmetry before running.

## Black-Litterman views

- Absolute views set an expected return for a single asset.
- Relative views set an expected outperformance between two assets.
- Confidence controls view strength.
- Tau controls prior uncertainty weighting.

## Feasibility guidance

Optimization failures are often caused by:

- minimum weights that are too high
- maximum weights that are too low
- too many forced weights
- overly restrictive linear constraints

The usual recovery path is to relax bounds, remove forced weights, and simplify constraints.

## Output tabs

### Weights

Allocation by asset over time, shown as chart or table.

### Turnover

Absolute allocation changes between rebalance windows.

### Statistics

Portfolio-level performance and risk metrics.

### Returns

Portfolio return time series.

### Growth of $1

Compounded wealth path from an initial value of one.

### Rolling

Trailing total return, volatility, Sharpe, and Sortino views across selected windows.

### Calendar Year

One-row-per-year compounded annual return table.

### Drawdown

Peak-to-trough drawdown paths as chart or table.

### Attribution

Asset-level contribution to portfolio returns.

### Risk

Asset-level contribution to portfolio risk over time.

### Frontier

Efficient frontier output with the active portfolio marker.

## Session management and export

- The portfolio dropdown selects the active saved result.
- Multiple named portfolios can be kept for scenario comparison.
- File > Save session exports JSON state.
- File > Load session restores it.
- File > Download Excel exports settings and tabular outputs for weights, turnover, statistics, returns, growth, rolling, calendar year, drawdown, attribution, risk, and frontier.
