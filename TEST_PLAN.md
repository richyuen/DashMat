# DashMat Test Plan

This document outlines the testing strategy for the DashMat application, covering environment setup, automated testing, and manual verification of all features.

## 1. Environment Setup

Before running tests, ensure the environment is correctly configured.

### 1.1. Installation
```bash
# Create and activate conda environment
conda create -n dashmat python=3.11 -y
conda activate dashmat

# Install dependencies
pip install -r requirements.txt
```

### 1.2. Test Data Generation
Generate synthetic test data for manual testing:
```bash
python generate_test_data.py
```
This creates `test_data.csv` in the root directory.

## 2. Automated Tests

### 2.1. Factor Risk Parity Validation
Run the factor risk parity test to ensure differentiation between risk models:
```bash
python test_factor_rp.py
```
**Pass Criteria:**
- Output confirms: "PASS: All three models produce different weight vectors"
- confirm L1 distances between models > 0

## 3. Manual Testing: Analytics Tool (`/analyticstool`)

Launch the application:
```bash
conda run -n dashmat python app.py --debug
```
 Navigate to `http://127.0.0.1:8050/analyticstool`.

### 3.1. Data Ingestion
- [ ] **Upload File**: Upload `test_data.csv`.
- [ ] **Percent Parsing**: Upload a file with `%` formatted cells (ensure they convert to decimals).
- [ ] **Date Parsing**: Verify dates are recognized and set as the index.

### 3.2. Series Management
- [ ] **Modal**: Open "Select Series" modal.
- [ ] **Selection**: Select variable number of series (1, 5, 10+).
- [ ] **Renaming**: Rename "Series_1" to "Equity".
- [ ] **Benchmark**: Assign "Series_10" as benchmark for "Series_1".
- [ ] **Ordering**: Use Up/Down buttons to reorder series.
- [ ] **Persistence**: Close and reopen modal; verify changes persist.

### 3.3. Analysis Controls
- [ ] **Periodicity**: Switch between Daily, Weekly (various days), and Monthly.
- [ ] **Returns Type**: Toggle between Total and Excess Returns.
- [ ] **Vol Scaling**: Set Vol Scaler to 10% and verify stats update.
- [ ] **Dates**: Filter by specific start/end dates.

### 3.4. Visualizations (Tabs)
- [ ] **Statistics**: Verify Sharpe, Sortino, and Max Drawdown calculations.
- [ ] **Returns**: Check data grid values.
- [ ] **Rolling**:
    - [ ] Test 3m, 6m, 1y windows.
    - [ ] Test Rolling Volatility and Beta.
- [ ] **Calendar Year**: Verify Heatmap and Monthly views.
- [ ] **Growth of $1**: Verify chart interactive elements.
- [ ] **Drawdown**: Check drawdown depth and duration.
- [ ] **Correlation**:
    - [ ] < 10 series: Verify Correlogram (Scatter Matrix).
    - [ ] > 10 series: Verify fallback to Heatmap only (performance check).

## 4. Manual Testing: Portfolio Optimization (`/portopt`)

Navigate to `http://127.0.0.1:8050/portopt`.

### 4.1. Optimization Configuration
- [ ] **Inputs**: Ensure series from Analytics Tool are available.
- [ ] **Model Selection**: Test all models:
    - [ ] Risk Parity
    - [ ] Factor Risk Parity
    - [ ] HRP
    - [ ] Max Sharpe
    - [ ] Min CVaR
    - [ ] Equal Weight
- [ ] **Parameters**:
    - [ ] Toggle "Exp Wt Cov" (Half-Life).
    - [ ] Test Rolling vs. Expanding windows.
    - [ ] Adjust Window Size (e.g., 126 vs 252).
    - [ ] Test "Fill In-Sample" ON/OFF.

### 4.2. Execution & Results
- [ ] **Run**: Execute optimization.
- [ ] **Progress**: Verify loading spinner/modal appears.
- [ ] **Completion**: Verify success message.
- [ ] **Tabs**:
    - [ ] **Weights**: Check allocation over time.
    - [ ] **Attribution**: Verify risk contribution charts.
    - [ ] **Statistics**: Compare portfolio stats vs components.
    - [ ] **Returns**: Verify portfolio return series.
    - [ ] **Growth**: Check cumulative performance chart.

### 4.3. Portfolio Management
- [ ] **Save**: Verify portfolio is saved to dropdown.
- [ ] **Compare**: Select multiple portfolios in "Compare" multiselect.
- [ ] **Delete**: Delete a generated portfolio.

## 5. System Health
- [ ] **Performance**: Verify no long lags (>2s) on tab switches.
- [ ] **Errors**: Check console (F12) for red javascript errors.
- [ ] **Export**: "File > Download Excel" generates valid .xlsx with all sheets.
