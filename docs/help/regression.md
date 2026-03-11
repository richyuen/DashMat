# Regression Analysis

Regression Analysis is the DashMat workspace for return-based model fitting, diagnostics, rolling estimation, and regression-derived performance views. It is designed for workflows where one series is treated as the dependent variable and one or more other series are treated as explanatory variables, factors, or style proxies.

Use this page when you want to answer questions such as:

- Which factors explain a strategy's return history?
- How stable are betas through time?
- Does a constrained or regularized model produce a more credible exposure profile?
- How do fitted results compare across rolling windows, diagnostics, and performance outputs?

## Typical workflow

1. Import the dependent series and the candidate explanatory series.
2. Open **Series Selection** and choose exactly one `Y` series and one or more `X` series.
3. Configure benchmark, long-short, volatility scaling, lags, and constrained beta settings where needed.
4. Set periodicity, date range, missing-data behavior, and model-specific options.
5. Choose the regression type and any advanced controls.
6. Run the model and save the result under a name.
7. Review ANOVA, coefficients, rolling outputs, derived returns, scatter diagnostics, and the residual Q-Q plot.
8. Save multiple named results if you want to compare model families or parameter settings.

## Data sources and import behavior

Regression supports several data paths, not just file upload.

### Supported source types

- AA Tool index imports can append one or more core categories.
- Portfolio imports support peer-relative, index-relative, and alternative portfolio streams.
- Raw imports support factor, funds, and performance workflows with staged rows before import.
- File imports support `CSV`, `XLS`, and `XLSX`, including multi-sheet workbooks.

### Practical import guidance

- Use unique series names across all import paths.
- Import all candidate regressors before configuring the model so the full universe is available in Series Selection.
- If a model fails unexpectedly, verify that the relevant series were actually imported under the names you think they were.

## Series Selection

The **Series Selection** modal defines the model inputs. This is the most important setup screen on the page.

### Required structure

- Exactly one series should be chosen as dependent variable `Y`.
- One or more series should be chosen as independent variables `X`.

### Additional row-level settings

- benchmark assignment
- long-short behavior
- volatility-scaling flags
- lag settings
- constrained beta min and max values

### Important rule for self-reference

If the same series is used as both `Y` and one of the `X` variables, its lag must be at least one period. Without the lag, the specification is not meaningful.

### Benchmark and long-short behavior

Benchmark assignment and long-short logic matter here for the same reason they matter on the analytics pages: they can transform the actual return stream used in estimation. If a dependent series is benchmark-relative under the current setup, interpret the regression results as explaining the transformed stream, not the original raw return series.

## Core controls and time settings

The top-level controls determine the sample and transformed data used in the regression.

### Periodicity

Choose from frequencies supported by the imported data and the allowed conversions. Changing periodicity changes:

- the estimation sample
- annualization conventions
- lag interpretation
- rolling-window meaning
- the scale of some diagnostics and performance outputs

### Vol Scaler

The global **Vol Scaler** applies to any series that has volatility scaling enabled in Series Selection. This is useful when comparing exposures on risk-normalized inputs, but it also changes the return history being modeled.

### Date range

Use explicit **Start Date** and **End Date** inputs or shortcut buttons such as **Common Range** and **Max Range**. Date range selection is especially important in regression because:

- factor availability often differs across series
- lagged regressors reduce the usable aligned sample
- rolling windows need sufficient history to produce stable outputs

### Missing data

The page supports options such as **Fill NA** and **Fill 0**. These are modeling choices, not visual preferences. Different missing-data settings can change coefficients materially.

**Fill in-sample** also affects how rolling and expanding workflows behave where forecasting-style treatment is relevant.

## Regression models

### OLS

**What it is:** standard unconstrained ordinary least squares.

**When to use it:** as the baseline model for coefficient interpretation and diagnostics.

**Typical use:** start here unless you already know that policy constraints or heavy multicollinearity require something else.

### Constrained OLS

**What it is:** OLS with per-variable beta limits and optional linear constraints.

**When to use it:** when factor exposures must stay within policy bands or economically sensible ranges.

**Common pitfall:** adding too many hard constraints can make the problem infeasible or distort the fit sharply.

### Style Analysis

**What it is:** constrained exposure decomposition where factor weights are bounded and typically sum to one.

**When to use it:** when the goal is to estimate style mix rather than unconstrained statistical fit.

**Common pitfall:** if the style proxy set is incomplete, the model may push too much explanatory power into the wrong factors.

### Ridge

**What it is:** `L2`-regularized regression that shrinks coefficients toward zero.

**When to use it:** when multicollinearity is high and coefficient stability matters more than sparsity.

**Common pitfall:** a large alpha can over-shrink economically meaningful exposures.

### Lasso

**What it is:** `L1`-regularized regression that can drive some coefficients to zero.

**When to use it:** when you want sparse factor selection from a larger candidate set.

**Common pitfall:** selection can be unstable when predictors are highly correlated.

### Elastic Net

**What it is:** a combination of `L1` and `L2` regularization.

**When to use it:** when predictors are correlated and you want both shrinkage and some sparsity control.

**Common pitfall:** alpha and `l1-ratio` should be interpreted together. Extreme settings collapse toward Ridge or Lasso behavior.

### ARIMA and GARCH residual overlays

**What they are:** residual models layered on top of OLS-family regressions.

**When to use them:** when residuals show serial correlation or volatility clustering.

**Common pitfall:** these overlays refine residual modeling; they do not fix a poor factor specification.

## Advanced model controls

### Intercept and standard-error options

- **Force Zero Intercept** is available where supported.
- **Robust SE** changes the standard-error treatment for diagnostics.

Force a zero intercept only when that restriction is economically justified. Otherwise it can distort coefficients to absorb the missing constant term.

### Exponential weighting

Exponential weighting uses **Exp Wt** plus **Half-Life** or lambda-style interpretation, depending on the control. Use it when you want recent observations to matter more heavily than older ones.

### Window controls

The page supports:

- **Full**
- **Expanding**
- **Rolling**

**Window Size** and **Opt Step** with **Unit** control how often the model is refreshed and how much history each estimation uses.

Rolling and expanding outputs are useful for checking whether exposures are stable or regime-dependent.

### Regularization parameters

- Ridge, Lasso, and Elastic Net expose **alpha**.
- Elastic Net also exposes **l1-ratio**.

Treat these as model-definition choices, not small tuning decorations. They can change both explanatory power and interpretability.

## Linear constraints

Constrained models can use linear beta constraints in addition to per-factor min and max beta bounds.

### Structure

- **Add Constraint** appends a row with coefficients plus **Min** and **Max** bounds.
- Blank rows are ignored safely.
- Each row enforces a condition of the form:

`Min <= sum(coef_i * beta_i) <= Max`

### When to use them

Use linear constraints when the rule involves a combination of exposures rather than a single beta. Examples include policy limits on grouped factor exposure or minimum required participation in a style bucket.

### Constraint guidance

- Start with the smallest set of necessary policy rules.
- Re-run after each new constraint.
- If the constrained model fails, relax or remove the most recent rule before changing unrelated settings.

## Output tabs

### ANOVA

The **ANOVA** output is the main diagnostic summary. It is the first place to inspect fit quality, significance, and overall model behavior.

### Rolling Summary and Rolling

These views show how model behavior evolves over time. Use them to detect:

- unstable exposures
- regime-dependent factor relationships
- performance that is strong only in a narrow sample

### Weights

The **Weights** tab shows coefficient or exposure behavior through time for rolling and expanding workflows. This tab is especially useful when the headline regression fit is acceptable but you need to know whether the factor mix is stable.

### Statistics and Returns

These tabs show performance metrics and transformed time-series outputs for the active regression result. They help connect the model back to a portfolio-style return interpretation instead of stopping at coefficients alone.

### Growth of $1, Calendar Year, and Drawdown

These views turn the selected regression result into performance-path outputs:

- compounded wealth path
- annual return table
- peak-to-trough drawdown history

They are useful when a model has attractive statistical diagnostics but unattractive realized path behavior.

### Scatter

The **Scatter** tab supports:

- residual versus predicted
- actual versus predicted
- comparisons against individual `X` variables

Use this tab to look for:

- outliers
- nonlinearity
- residual structure
- variables that appear visually weak despite statistical inclusion

## Result management, sessions, and export

- Results are saved by name and can be reselected later from the result dropdown.
- **Save session** exports current session storage to JSON.
- **Load session** restores page state from a saved JSON.
- **New session** clears session state and reloads the page.
- **Download Excel** exports summary, coefficients, diagnostics, predicted and residual outputs, plus tabular outputs for returns, growth, rolling, calendar year, and drawdown.

If you are comparing model variants, use descriptive result names that encode the model family and the most important tuning parameter, such as window type or alpha.

## Common mistakes and troubleshooting

### Run fails immediately

First verify:

- exactly one `Y`
- at least one `X`
- sufficient overlapping date coverage
- no invalid self-reference without lag

### Coefficients look implausible

Check multicollinearity, lag choices, missing-data treatment, and whether the dependent series was transformed through benchmark-relative or long-short settings.

### Constrained models will not solve

The most common cause is an over-constrained specification. Relax beta bounds or simplify the linear constraint set.

### Rolling outputs are sparse or empty

The current window size may be too large for the available aligned history, especially after lags are applied.

### Periodicity options do not look right

Verify the source frequency of the imported data and whether the current session already contains converted data at another periodicity.

### Diagnostics are statistically fine but economically unconvincing

That usually means the factor set, sample period, or model family needs reconsideration. Good fit statistics do not guarantee a meaningful economic story.
