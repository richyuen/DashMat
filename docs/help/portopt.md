# Portfolio Optimization

Portfolio Optimization is the DashMat workspace for building, saving, and comparing portfolio allocations. It supports historical optimization, risk-based methods, ex-ante mean-variance setups, and Black-Litterman workflows. The page is intended for users who want to move from a collection of asset return series to a named portfolio result with weights, turnover, performance, attribution, risk, and frontier outputs.

This page is the right tool when you need to answer questions such as:

- Which allocation would maximize risk-adjusted return under a set of bounds?
- How sensitive is a portfolio to my window choice, rebalance cadence, or missing-data handling?
- What happens if I impose grouped exposure constraints?
- How do ex-ante views and Black-Litterman assumptions change the resulting allocation?

## Typical workflow

1. Import return series and decide which ones represent portfolio assets.
2. Open **Series Selection** and configure the asset universe.
3. Set top-level preprocessing controls such as periodicity, date range, and volatility scaling.
4. Choose the optimization model.
5. Configure estimation controls such as window type, step size, missing-data handling, and optional exponential weighting.
6. Add per-asset bounds, force-max flags, linear constraints, or ex-ante assumptions if needed.
7. Enter a portfolio name and run the optimization.
8. Review the result across weights, turnover, returns, statistics, attribution, risk, and frontier tabs.
9. Save multiple named portfolios if you want to compare scenarios.

## Data requirements and import behavior

### Supported inputs

- Input files should be date-indexed return series with one column per asset.
- Supported upload formats are `CSV`, `XLS`, and `XLSX`.
- Workbook imports can include one or more selected sheets.
- Values may be stored as decimals or percentage-style values.

### Frequency rules

- Periodicity is auto-detected on upload.
- Daily data can be resampled to weekly or monthly.
- Monthly data is not upsampled to daily.
- Available history and date overlap across selected assets determine whether some optimizations and tabs have enough data to render.

### Practical import guidance

- Use clean, unique asset names before upload.
- Avoid mixing unrelated assets into the working universe unless they are genuinely candidates for optimization.
- Review the imported data before running an optimizer. Most optimization failures are not solver issues; they are specification issues caused by thin, noisy, or inconsistent inputs.

## Series Selection

The **Series Selection** modal defines the optimization universe and the hard per-asset rules.

### Core row-level controls

- **Include** determines whether the asset participates in optimization.
- Drag and drop sets display order in selectors and outputs.
- **Benchmark** assigns a comparison series for excess-return and long-short behavior.
- **Long-Short** transforms a series into `series - benchmark`.
- **Scale Vol** enables use of the page-level **Vol Scaler** for that asset.
- **Min Wt** and **Max Wt** are hard optimizer bounds.
- **Force Max** pins the asset to its maximum weight.

### Why this modal matters

This modal does more than clean up display order. It defines the feasible region for the optimization. A run can fail or produce a surprising allocation because:

- too many assets were excluded
- a benchmark-relative transformation changed the asset return series
- min weights add up to more than `100%`
- max weights are too tight to allow a valid solution
- force-max pins too many assets and leaves no room for the rest

### Force Max guidance

**Force Max** is powerful but easy to misuse. It effectively says that the asset must sit at its maximum allocation. If you apply that to several assets at once, or combine it with restrictive linear constraints, you can eliminate the feasible region entirely.

## Top-level controls and preprocessing

The controls above the result tabs determine the return series the optimizer actually sees.

### Periodicity

**Periodicity** converts returns to the working frequency used in optimization and downstream reporting. It is not just a display preference. It changes:

- the sample used to estimate parameters
- annualization conventions
- rolling and rebalance behavior
- the shape of performance and risk outputs

### Vol Scaler

**Vol Scaler** applies a target annualized volatility level to any asset with **Scale Vol** enabled in Series Selection. Set it to zero to disable scaling.

Common uses:

- putting strategies with different realized risk on a comparable footing
- stress-testing how sensitive the optimizer is to leverage-like scaling
- comparing the effect of risk normalization on model rankings

### Date range

Use explicit date inputs or the shortcut buttons:

- **Common Range** uses only dates shared by all selected assets.
- **Common Daily** moves to the daily overlap and sets periodicity to daily.
- **Max Range** uses the earliest start and latest end across the selected assets.

The chosen date range matters twice: first for estimation, and then again for displayed performance of the optimized result.

## Optimization controls

These controls define how the optimization is estimated and how often it is refreshed.

### Portfolio Name

The portfolio name is the saved result key. Give distinct names to materially different scenarios so you can compare them later rather than overwriting a result you still want to inspect.

### Model

The model selector chooses the optimization family. Some models use historical estimates only, while others require additional ex-ante inputs or views.

### Exponential weighting

**Exp Wt** enables exponentially weighted historical estimates. Decay values are interpreted as:

- half-life periods when the value is greater than or equal to `1`
- lambda when the value is below `1`

Use exponential weighting when you want more recent observations to matter more than older ones.

### Window controls

The page supports:

- **Full**: use the full available sample
- **Expanding**: grow the estimation window through time
- **Rolling**: use a fixed-length lookback window

**Window Size** controls the lookback length for rolling behavior. **Opt Step** and **Unit** control how often the optimizer rebalances and records a new portfolio.

### Missing data

The missing-data options such as **Fill NA** and **Fill 0** are not cosmetic. They affect the estimation sample and therefore the resulting allocation. Use them carefully:

- filling with zeros may be reasonable for explicit no-exposure periods in some constructed series
- filling with zeros may be misleading for genuinely missing market returns
- leaving gaps unhandled can reduce usable sample size or block some windows entirely

## Model guide

### Risk Parity

**What it does:** balances total risk contribution across assets rather than chasing an explicit expected-return forecast.

**When to use it:** when return forecasts are weak or unstable and diversification of risk is the primary objective.

**Common pitfalls:** very tight bounds or poor covariance estimates can still force concentrated outcomes.

### Factor Risk Parity

**What it does:** balances risk through latent factor structure rather than only pairwise covariance.

**When to use it:** when the assets share common drivers and you want a more structured notion of diversification.

**Common pitfalls:** if the factor structure implied by the sample is noisy, the result may still be unstable.

### Hierarchical Risk Parity

**What it does:** clusters assets and allocates top-down.

**When to use it:** when correlation estimates are noisy and clustering provides a more robust diversification framework.

**Common pitfalls:** HRP is not immune to poor input data; cluster structure can still shift meaningfully with sample changes.

### Maximize Sharpe Ratio

**What it does:** seeks the highest expected return per unit of volatility.

**When to use it:** when you want a classical return-risk efficient allocation from historical or supplied estimates.

**Common pitfalls:** mean-return estimates are noisy, so this model can overreact to recent winners or outliers.

### Minimize Variance

**What it does:** seeks the lowest-volatility portfolio under the current constraints.

**When to use it:** when capital preservation, smoothness, or diversification is more important than maximizing expected return.

**Common pitfalls:** the optimizer may allocate heavily to low-volatility assets in ways that are mathematically valid but economically too defensive.

### Minimize CVaR

**What it does:** targets downside tail risk rather than variance alone.

**When to use it:** when tail behavior and loss severity matter more than symmetric volatility.

**Common pitfalls:** tail estimates require enough useful history. Thin samples can make this model unstable.

### Equal Weight

**What it does:** applies a simple `1/N` allocation baseline, subject to whatever hard constraints still apply.

**When to use it:** as a neutral benchmark or when estimation uncertainty is so high that a simple baseline is preferable to a fragile optimized result.

**Common pitfalls:** equal weight is simple, not automatically feasible under every custom bound and constraint set.

### Ex Ante Mean-Variance

**What it does:** uses user-supplied expected returns and risk assumptions directly rather than relying only on historical estimates.

**When to use it:** when you have forward-looking views or policy assumptions that should drive the portfolio.

**Common pitfalls:** unrealistic returns, impossible volatilities, inconsistent correlations, or non-symmetric matrices can invalidate the run or make the result economically meaningless.

### Black-Litterman

**What it does:** combines prior assumptions with user views and confidence levels to produce posterior return and covariance inputs.

**When to use it:** when you want a structured way to blend strategic priors with tactical views.

**Common pitfalls:** very strong confidence values or weakly specified priors can dominate the result in ways that are mathematically consistent but not intended.

## Linear constraints

Linear constraints allow grouped exposure rules that cannot be expressed as simple per-asset min and max bounds.

Each constraint row has:

- a **Min** bound
- a **Max** bound
- one coefficient column per selected asset

The row enforces:

`Min <= sum(coef_i * weight_i) <= Max`

### Example

If equities have coefficient `1`, credit has coefficient `1`, and all other assets have coefficient `0`, then a row with `Min = 0.40` and `Max = 0.70` constrains the combined weight of those two groups to stay between `40%` and `70%`.

### Constraint design guidance

- Start simple and add one policy band at a time.
- Re-run after each major constraint addition.
- If a feasible portfolio suddenly disappears, the newest constraint is the first place to check.

## Ex-ante inputs

Ex Ante Mean-Variance and Black-Litterman require forward-looking assumptions.

### Input modes

The page supports two main forms:

- expected return plus covariance
- expected return plus volatility plus correlation

Use whichever form is easier to source and validate, but confirm that the implied matrix is economically reasonable.

### Good practice

- Use the upload and estimate helpers to seed the grid.
- Check that volatilities are on the intended scale.
- Check that the covariance or correlation structure is symmetric.
- Confirm that correlations lie in sensible ranges and that the overall matrix is not obviously contradictory.

## Black-Litterman views

Black-Litterman allows both **absolute** and **relative** views.

- **Absolute view**: set an expected return for one asset.
- **Relative view**: set an expected outperformance between two assets.

### Examples

- Absolute: if `SPY` is entered with expected return `3`, the view says `SPY` should earn `3%`.
- Relative: if `QQQ` is set to outperform `SPY` by `2`, the view says the spread `QQQ - SPY` should be `2%`.

### Confidence and tau

- **Confidence** controls how strongly a view influences the posterior result.
- **Tau** controls the uncertainty weighting of the prior.

These are powerful levers. Extreme settings can move the optimizer much more than expected.

## Feasibility and failure recovery

Optimization failures are usually caused by specification conflicts, not by the solver randomly breaking.

Common causes:

- minimum weights set too high
- maximum weights set too low
- too many assets forced to maximum
- overly restrictive linear constraints
- inconsistent ex-ante assumptions
- insufficient usable history after date filtering and missing-data handling

When a run fails, the usual recovery order is:

1. Relax or remove **Force Max**.
2. Check whether min weights and max weights are jointly feasible.
3. Remove the newest linear constraint.
4. Simplify ex-ante assumptions or views.
5. Widen the usable sample if the history is too thin.

## Output tabs

### Weights

The **Weights** tab shows the portfolio allocation by asset over time, either as a table or chart. Use it to see concentration, turnover drivers, and how the optimizer responds to changing windows.

### Turnover

The **Turnover** tab shows the absolute change in allocation between rebalance windows. High turnover can indicate model sensitivity, unstable estimates, or an excessively short rebalance cycle.

### Statistics

The **Statistics** tab summarizes portfolio-level performance and risk. It is the quickest way to compare saved portfolios after confirming that they were run on comparable settings.

### Returns

The **Returns** tab shows the portfolio return series itself. Check this tab if a chart looks wrong or if you need to reconcile the portfolio with another export.

### Growth of $1

This tab compounds the portfolio return stream into a wealth path. It is useful for communicating how different models feel through time, not just where they end.

### Rolling

The **Rolling** tab shows trailing return and risk behavior across selected windows. Use it to judge stability rather than relying only on full-sample metrics.

### Calendar Year

This tab summarizes one compounded return per year. It is useful for spotting sequencing risk and years where a model behaved very differently from its long-run average.

### Drawdown

The **Drawdown** tab shows peak-to-trough loss paths. It helps distinguish portfolios that have similar total returns but very different pain profiles.

### Attribution

The **Attribution** tab shows asset-level contribution to portfolio returns. It is the first place to look when a portfolio outperformed but the reason is not obvious from the weights alone.

### Risk

The **Risk** tab shows asset-level contribution to portfolio risk through time. This is particularly useful when headline weights look balanced but realized risk is still concentrated.

### Frontier

The **Frontier** tab shows the efficient frontier and the active portfolio marker. Use it to understand whether the chosen portfolio sits on a reasonable part of the opportunity set and how sensitive the trade-off looks under the current assumptions.

## Saved portfolios, session management, and export

- The portfolio dropdown selects the active saved result.
- Multiple named portfolios can be retained for scenario comparison.
- **File > Save session** exports page state to JSON.
- **File > Load session** restores it.
- **File > Download Excel** exports settings and tabular outputs for weights, turnover, statistics, returns, growth, rolling, calendar year, drawdown, attribution, risk, and frontier.

If you are comparing scenarios, use clear portfolio names that encode the main decision variable, such as model family, window type, or constraints.

## Common mistakes and troubleshooting

### The optimizer fails immediately

Check feasibility first: min weights, max weights, force-max flags, and linear constraints.

### The results look too concentrated

That may be a rational response to the chosen objective and estimates. Review the model choice, bounds, window, and whether a recent outlier is dominating historical means or covariances.

### Frontier or risk outputs look inconsistent with weights

Remember that risk contribution is not the same thing as capital weight. A small capital weight can still contribute a large amount of risk.

### Ex-ante models produce strange results

Recheck the assumptions grid before changing the optimizer. Many surprising allocations come from input scaling, correlation mistakes, or overly strong Black-Litterman views.

### Turnover is too high

Increase the rebalance step, use a longer window, simplify the model, or inspect whether the input data is too noisy for the chosen cadence.
