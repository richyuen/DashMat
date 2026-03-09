# Regression Analysis

Regression Analysis is the DashMat workspace for return-based regressions, diagnostics, rolling windows, and regression-oriented performance outputs. Use it to assign dependent and independent variables, fit several regression model types, and review coefficients, diagnostics, rolling summaries, and derived return views.

## Typical workflow

1. Load data from file, database-backed imports, or existing sources.
2. Open Series Selection and choose one dependent variable and one or more independent variables.
3. Configure benchmark, long-short, volatility scaling, lags, and any constrained-beta settings.
4. Set periodicity, date range, missing-data handling, and model-specific controls.
5. Run the regression and save the result under a name.
6. Review ANOVA, rolling, weights, statistics, returns, growth, drawdown, and scatter outputs.

## Data sources and import

- AA Tool indices support appending one or more core categories.
- Portfolio imports support peer-relative, index-relative, and alternative portfolio streams.
- Raw imports support factor, funds, and performance workflows with staged rows before import.
- File imports support CSV, XLS, and XLSX, including multi-sheet selection.
- Duplicate series names are blocked across import paths.

## Series Selection

- Exactly one series should be chosen as the dependent variable `Y`.
- One or more series should be chosen as independent variables `X`.
- Optional benchmark, long-short, and per-series volatility scaling settings are available.
- Per-series lag and constrained beta bounds can also be configured.
- If `Y` is also selected as an `X`, its lag must be at least one period.

## Controls and time settings

### Periodicity

Choose from frequencies supported by the loaded data and any permitted conversions.

### Vol scaler

Apply a global volatility scaling percentage to series that have scaling enabled.

### Date range

Use explicit Start Date and End Date inputs, plus Common Range and Max Range shortcuts, to control the estimation sample.

### Missing data

Choose Fill NA or Fill 0. Fill in-sample controls how rolling and expanding workflows treat forecasting behavior where applicable.

## Regression models

### OLS

Baseline linear regression with unconstrained coefficients. Use it as the reference model for coefficient interpretation and diagnostics.

### Constrained OLS

OLS with per-variable beta limits and optional linear constraints. Use it when exposures must stay within policy bounds.

### Style Analysis

Constrained exposure decomposition where factor weights are bounded and sum to one. Use it to estimate style mix.

### Ridge

L2-regularized regression that shrinks coefficients but usually keeps all predictors. Use it for collinearity and coefficient stability.

### Lasso

L1-regularized regression that can push some coefficients to zero. Use it for sparse models and variable selection.

### Elastic Net

Combines L1 and L2 regularization. Use it when factors are correlated and you want both shrinkage and sparsity control.

### ARIMA and GARCH residual overlay

Residual-model overlays for OLS-family regressions. Use them when residuals show serial correlation or volatility clustering.

## Advanced model controls

- Force Zero Intercept and Robust SE are available where supported by model choice.
- Exponential weighting uses Exp Wt plus Half-Life.
- Window controls support Full, Expanding, and Rolling modes.
- Window Size and Opt Step with Unit control the estimation and reporting cadence.
- Ridge, Lasso, and Elastic Net expose alpha, and Elastic Net also exposes l1-ratio.

## Linear constraints

- Add Constraint appends rows with coefficients plus Min and Max bounds.
- Blank linear-constraint rows are ignored safely.
- If constrained models fail, relax bounds or simplify the constraint set.

## Output tabs

### ANOVA

Shows regression summary information and core diagnostics.

### Rolling Summary and Rolling

Shows rolling model behavior and rolling metrics such as total return, volatility, Sharpe Ratio, and Sortino Ratio where supported.

### Weights

Shows coefficient or exposure behavior over time for rolling and expanding workflows.

### Statistics and Returns

Shows performance metrics and transformed time-series outputs for the selected regression result.

### Growth of $1, Calendar Year, and Drawdown

Provides compounded wealth, annual return, and downside-path views for the active result stream.

### Scatter

Supports residual-versus-predicted, actual-versus-predicted, and X-variable comparison views.

## Session management and export

- Results are saved by name and can be selected again later from the result dropdown.
- Save session exports current session storage to JSON.
- Load session restores page state from a saved JSON.
- New session clears session state and reloads the page.
- Download Excel exports summary, coefficients, diagnostics, predicted and residual outputs, plus tabular outputs for returns, growth, rolling, calendar year, and drawdown.

## Troubleshooting

- If Run fails, first verify that `Y`, `X`, and date coverage are valid.
- If imports fail, check duplicate names, staged rows, and source availability.
- If periodicity options look incorrect, verify the original source frequency and current session state.
