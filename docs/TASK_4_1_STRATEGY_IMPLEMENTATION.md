# Task 4.1: Strategy Implementation

## Overview

This document describes the 5/15 EMA crossover strategy with regime filter implementation. The strategy combines technical analysis (EMA crossovers) with regime detection (HMM) to trade only in trending markets.

---

## Strategy Rules

### LONG Entry
- **Condition 1**: EMA(5) crosses ABOVE EMA(15) (bullish crossover)
- **Condition 2**: Current regime = +1 (Uptrend)
- **Execution**: Enter at NEXT candle open

### LONG Exit
- **Condition**: EMA(5) crosses BELOW EMA(15) (bearish crossover)
- **Execution**: Exit at NEXT candle open

### SHORT Entry
- **Condition 1**: EMA(5) crosses BELOW EMA(15) (bearish crossover)
- **Condition 2**: Current regime = -1 (Downtrend)
- **Execution**: Enter at NEXT candle open

### SHORT Exit
- **Condition**: EMA(5) crosses ABOVE EMA(15) (bullish crossover)
- **Execution**: Exit at NEXT candle open

### Regime Filter
- **No trades in Regime 0 (Sideways)**
- Regime is forward-filled to all 5-minute bars

---

## Strategy Logic Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY LOGIC                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐                                        │
│  │ EMA(5) crosses  │                                        │
│  │ above EMA(15)   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐     YES    ┌─────────────────┐        │
│  │ Regime = +1 ?   │──────────→ │ LONG ENTRY      │        │
│  └────────┬────────┘            │ (next open)     │        │
│           │ NO                  └─────────────────┘        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ No Trade        │                                        │
│  │ (Sideways/Down) │                                        │
│  └─────────────────┘                                        │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ EMA(5) crosses  │                                        │
│  │ below EMA(15)   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐     YES    ┌─────────────────┐        │
│  │ Regime = -1 ?   │──────────→ │ SHORT ENTRY     │        │
│  └────────┬────────┘            │ (next open)     │        │
│           │ NO                  └─────────────────┘        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ No Trade        │                                        │
│  │ (Sideways/Up)   │                                        │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Backtest Results

### Trade Summary

| Metric | Value |
|--------|-------|
| **Total Trades** | 364 |
| Long Trades | 194 |
| Short Trades | 170 |
| Winning Trades | 112 |
| Losing Trades | 252 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Win Rate** | 30.77% |
| **Total P&L** | 79.90 points |
| **Total P&L %** | -0.25% |
| **Avg P&L per Trade** | 0.22 points |
| **Profit Factor** | 1.01 |

### Best/Worst Trades

| Metric | Value |
|--------|-------|
| Max Win | 377.96 points |
| Max Loss | -115.00 points |
| Avg Win | 77.84 points |
| Avg Loss | -34.28 points |

### Long vs Short Performance

| Direction | Total P&L | Win Rate |
|-----------|-----------|----------|
| **Long** | -208.87 points | 30.93% |
| **Short** | +288.77 points | 30.59% |

---

## Key Observations

### 1. Low Win Rate, Positive Expectancy
- Win rate is only 30.77%, but strategy is slightly profitable
- Average win (77.84) is larger than average loss (34.28)
- Risk-reward ratio ≈ 2.3:1

### 2. Short Trades Outperform
- Short trades: +288.77 points
- Long trades: -208.87 points
- Market had more profitable short opportunities

### 3. Regime Filter Effect
- Without regime filter: 738 potential long + 738 potential short = 1,476 signals
- With regime filter: 194 long + 170 short = 364 trades
- Filter reduced trades by 75%, focusing on trending periods

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `nifty_features_5min.csv` | `data/processed/` | Dataset with signal columns |
| `trade_log.csv` | `data/processed/` | Detailed trade log |

### New Columns Added

| Column | Description |
|--------|-------------|
| `signal` | Trade signal (1=long, -1=short, 0=none) |
| `signal_type` | Signal type (LONG_ENTRY, SHORT_ENTRY, etc.) |
| `position` | Current position (1, -1, 0) |
| `trade_id` | Trade identifier |
| `regime_filled` | Forward-filled regime |

---

## Trade Log Structure

| Column | Description |
|--------|-------------|
| `trade_id` | Unique trade identifier |
| `entry_time` | Entry timestamp |
| `exit_time` | Exit timestamp |
| `entry_price` | Entry price (next candle open) |
| `exit_price` | Exit price (next candle open) |
| `type` | LONG or SHORT |
| `regime` | Regime at entry |
| `pnl` | Profit/Loss in points |
| `pnl_pct` | Profit/Loss percentage |
| `duration_bars` | Trade duration in bars |

---

## Sample Trades

```
Trade 1: LONG at 19339.95 → Exit at 19337.76, P&L: -2.19 (16 bars)
Trade 2: LONG at 19419.61 → Exit at 19382.14, P&L: -37.47 (1 bar)
Trade 3: LONG at 19406.65 → Exit at 19494.52, P&L: +87.87 (19 bars)
Trade 7: LONG at 19446.58 → Exit at 19573.82, P&L: +127.24 (42 bars)
```

---

## How to Use

### Running the Strategy

```bash
python run_strategy.py
```

### Accessing Results

```python
import pandas as pd

# Load signals
df = pd.read_csv('data/processed/nifty_features_5min.csv')

# Load trade log
trades = pd.read_csv('data/processed/trade_log.csv')

# Filter winning trades
winners = trades[trades['pnl'] > 0]

# Filter long trades
longs = trades[trades['type'] == 'LONG']
```

### Custom Backtest

```python
from src.strategy.ema_regime_strategy import EMARegimeStrategy

strategy = EMARegimeStrategy(fast_period=5, slow_period=15)
df_signals, trades = strategy.backtest(df)
stats = strategy.calculate_statistics(trades)
```

---

## Strategy Improvements (Future)

1. **Position Sizing**: Add volatility-based position sizing
2. **Stop Loss**: Add trailing stop or fixed stop loss
3. **Take Profit**: Add profit targets
4. **Time Filter**: Avoid trading during low-volume periods
5. **Regime Confidence**: Only trade when regime probability > 80%

---

## Glossary

| Term | Definition |
|------|------------|
| **EMA** | Exponential Moving Average |
| **Crossover** | When fast EMA crosses slow EMA |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **P&L** | Profit and Loss |
| **Regime Filter** | Only trade in specific market conditions |
