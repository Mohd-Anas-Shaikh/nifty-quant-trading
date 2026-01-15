# Task 4.2: Backtesting

## Overview

This document describes the comprehensive backtesting performed on the 5/15 EMA crossover strategy with regime filter. The backtest uses a 70/30 train/test split to evaluate out-of-sample performance.

---

## Data Split

| Period | Bars | Percentage | Date Range |
|--------|------|------------|------------|
| **Training** | 13,938 | 70% | 2025-01-15 to 2025-09-29 |
| **Testing** | 5,974 | 30% | 2025-09-29 to 2026-01-15 |
| **Full** | 19,912 | 100% | 2025-01-15 to 2026-01-15 |

---

## Performance Metrics Explained

### 1. Total Return
The percentage gain or loss from initial capital.
```
Total Return = (Final Value - Initial Value) / Initial Value × 100
```

### 2. Sharpe Ratio
Risk-adjusted return measuring excess return per unit of volatility.
```
Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
```
- **> 1.0**: Good
- **> 2.0**: Very Good
- **< 0**: Negative risk-adjusted return

### 3. Sortino Ratio
Like Sharpe but only penalizes downside volatility.
```
Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
```
- Higher is better
- More relevant for strategies with asymmetric returns

### 4. Calmar Ratio
Return relative to maximum drawdown risk.
```
Calmar = Annualized Return / Maximum Drawdown
```
- **> 1.0**: Good risk/reward
- **> 3.0**: Excellent

### 5. Maximum Drawdown
Largest peak-to-trough decline in portfolio value.
```
Max DD = (Peak - Trough) / Peak × 100
```
- Lower is better
- Critical for risk management

---

## Results Summary

### Training Period (70%)

| Metric | Value |
|--------|-------|
| **Total Return** | -0.52% (-515.73 points) |
| **Sharpe Ratio** | -5.77 |
| **Sortino Ratio** | -2.81 |
| **Calmar Ratio** | -0.71 |
| **Max Drawdown** | 0.98% |
| **Total Trades** | 259 |
| **Win Rate** | 28.19% |
| **Profit Factor** | 0.92 |
| **Avg Duration** | 11.8 bars |

### Testing Period (30%)

| Metric | Value |
|--------|-------|
| **Total Return** | +0.60% (+595.63 points) |
| **Sharpe Ratio** | -3.40 |
| **Sortino Ratio** | -1.77 |
| **Calmar Ratio** | 3.61 |
| **Max Drawdown** | 0.53% |
| **Total Trades** | 105 |
| **Win Rate** | 37.14% |
| **Profit Factor** | 1.25 |
| **Avg Duration** | 69.9 bars |

### Full Period

| Metric | Value |
|--------|-------|
| **Total Return** | +0.08% (+79.90 points) |
| **Sharpe Ratio** | -5.00 |
| **Sortino Ratio** | -2.49 |
| **Calmar Ratio** | 0.07 |
| **Max Drawdown** | 1.11% |
| **Total Trades** | 364 |
| **Win Rate** | 30.77% |
| **Profit Factor** | 1.01 |
| **Avg Duration** | 66.9 bars |

---

## Key Observations

### 1. Out-of-Sample Performance Improved
- Training: -0.52% return, 0.92 profit factor
- Testing: +0.60% return, 1.25 profit factor
- **Strategy performed better on unseen data**

### 2. Win Rate vs Risk-Reward
- Low win rate (~30%) but positive expectancy
- Avg Win (77.84) > 2× Avg Loss (34.28)
- Classic trend-following characteristic

### 3. Drawdown Control
- Max drawdown only 1.11% on full period
- Conservative risk profile
- Regime filter helps avoid large losses

### 4. Trade Duration
- Training: 11.8 bars average
- Testing: 69.9 bars average
- Longer trades in test period (more trending)

### 5. Negative Sharpe Ratios
- Due to low absolute returns relative to risk-free rate (6.5%)
- Strategy is marginally profitable but not beating risk-free

---

## Trade Breakdown

### By Direction

| Period | Long Trades | Short Trades | Long Win Rate | Short Win Rate |
|--------|-------------|--------------|---------------|----------------|
| Training | 160 | 99 | ~28% | ~28% |
| Testing | 34 | 71 | ~37% | ~37% |
| Full | 194 | 170 | 30.93% | 30.59% |

### By Outcome

| Period | Winners | Losers | Win Rate |
|--------|---------|--------|----------|
| Training | 73 | 186 | 28.19% |
| Testing | 39 | 66 | 37.14% |
| Full | 112 | 252 | 30.77% |

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `backtest_report.txt` | `data/processed/` | Detailed text report |
| `equity_curve.csv` | `data/processed/` | Portfolio value over time |

---

## Metrics Formulas

### Sharpe Ratio (Annualized)
```python
sharpe = (mean_return * periods_per_year - risk_free_rate) / (std_return * sqrt(periods_per_year))
```

### Sortino Ratio (Annualized)
```python
downside_returns = returns[returns < 0]
sortino = (mean_return * periods_per_year - risk_free_rate) / (downside_std * sqrt(periods_per_year))
```

### Calmar Ratio
```python
annualized_return = (1 + total_return) ^ (1/years) - 1
calmar = annualized_return / max_drawdown
```

### Profit Factor
```python
profit_factor = gross_profit / gross_loss
```

---

## Interpretation

### Strategy Assessment

| Aspect | Assessment |
|--------|------------|
| **Profitability** | Marginally profitable (+0.08% overall) |
| **Risk-Adjusted** | Below risk-free rate (negative Sharpe) |
| **Drawdown** | Well controlled (<1.5%) |
| **Consistency** | Better on test than train (good sign) |
| **Scalability** | Low returns may not justify transaction costs |

### Recommendations

1. **Add Position Sizing**: Volatility-based sizing could improve Sharpe
2. **Optimize Parameters**: Test different EMA periods
3. **Add Filters**: Time-of-day, volatility filters
4. **Risk Management**: Add stop-loss to reduce losing trades

---

## How to Run

```bash
python run_backtest.py
```

### Custom Backtest

```python
from src.strategy.backtester import Backtester, EMARegimeStrategy

backtester = Backtester(train_ratio=0.7, initial_capital=100000)
strategy = EMARegimeStrategy(fast_period=5, slow_period=15)

train_result, test_result, full_result = backtester.run_backtest(df, strategy)
backtester.print_results()
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Risk-adjusted return using downside volatility only |
| **Calmar Ratio** | Return / Maximum Drawdown |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Profit Factor** | Gross profits / Gross losses |
| **Win Rate** | Percentage of profitable trades |
| **Risk-Free Rate** | Return on risk-free investment (6.5% used) |
