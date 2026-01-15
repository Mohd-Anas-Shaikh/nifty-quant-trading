# Task 2.1: EMA Indicators

## Overview

This document explains the Exponential Moving Average (EMA) indicators created for our trading system. These EMAs will be used exclusively for generating trading signals.

---

## What is EMA?

**Exponential Moving Average (EMA)** is a type of moving average that gives more weight to recent prices, making it more responsive to new information.

### EMA vs SMA (Simple Moving Average)

Think of it like this:
- **SMA** is like asking 5 friends their opinion and giving each equal weight
- **EMA** is like asking 5 friends but trusting your most recent conversations more

```
┌─────────────────────────────────────────────────────────────┐
│                    SMA vs EMA Comparison                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day:        1     2     3     4     5                      │
│  Price:     100   102   104   103   105                     │
│                                                              │
│  SMA(5) = (100 + 102 + 104 + 103 + 105) / 5 = 102.8        │
│           Equal weight to all prices                        │
│                                                              │
│  EMA(5) = More weight to recent prices                      │
│           Day 5 has highest weight, Day 1 has lowest        │
│           Result ≈ 103.5 (closer to recent price)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## EMA Formula

```
EMA_today = Price_today × k + EMA_yesterday × (1 - k)

Where:
  k = 2 / (period + 1)    ← "smoothing factor" or "alpha"
```

### Our EMAs

| EMA | Period | Alpha (k) | Purpose |
|-----|--------|-----------|---------|
| **Fast EMA** | 5 | 0.3333 | Captures short-term momentum |
| **Slow EMA** | 15 | 0.1250 | Captures medium-term trend |

**Why these periods?**
- **EMA(5)**: Reacts quickly to price changes (5 × 5min = 25 minutes of data)
- **EMA(15)**: Smoother, filters out noise (15 × 5min = 75 minutes of data)

---

## Trading Signals

### EMA Crossover Strategy

The relationship between fast and slow EMA generates trading signals:

```
┌─────────────────────────────────────────────────────────────┐
│                    EMA Crossover Signals                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BULLISH CROSSOVER (Buy Signal)                             │
│  ─────────────────────────────                              │
│  Fast EMA crosses ABOVE Slow EMA                            │
│                                                              │
│       Slow EMA ──────────────                               │
│                      ╲                                       │
│                       ╲ ← Crossover point                   │
│                        ╲                                     │
│       Fast EMA ─────────────────                            │
│                                                              │
│  Interpretation: Short-term momentum turning positive       │
│                                                              │
│  ───────────────────────────────────────────────────────    │
│                                                              │
│  BEARISH CROSSOVER (Sell Signal)                            │
│  ─────────────────────────────                              │
│  Fast EMA crosses BELOW Slow EMA                            │
│                                                              │
│       Fast EMA ──────────────                               │
│                      ╲                                       │
│                       ╲ ← Crossover point                   │
│                        ╲                                     │
│       Slow EMA ─────────────────                            │
│                                                              │
│  Interpretation: Short-term momentum turning negative       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Trend Direction

| Condition | Trend | Meaning |
|-----------|-------|---------|
| Fast EMA > Slow EMA | Uptrend (+1) | Prices rising, bullish |
| Fast EMA < Slow EMA | Downtrend (-1) | Prices falling, bearish |

---

## Features Created

| Column | Description | Values |
|--------|-------------|--------|
| `ema_fast` | 5-period EMA | Price level |
| `ema_slow` | 15-period EMA | Price level |
| `ema_diff` | Fast EMA - Slow EMA | Points (+ or -) |
| `ema_diff_pct` | Difference as percentage | % |
| `ema_crossover` | Crossover signal | 1, -1, or 0 |
| `ema_trend` | Current trend direction | 1 or -1 |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Total Records | 19,912 |
| Bullish Crossovers | 738 |
| Bearish Crossovers | 738 |
| Time in Uptrend | 52.34% |
| Time in Downtrend | 47.66% |
| Avg EMA Difference | 1.31 points |

**Interpretation:**
- Market spent slightly more time in uptrend (52.34%)
- Equal number of bullish and bearish crossovers (balanced market)
- Average EMA difference is small (1.31 points), indicating frequent trend changes

---

## Sample Data

```
timestamp            spot_close  ema_fast    ema_slow    ema_diff   crossover  trend
2025-01-15 09:15:00  19510.69    19510.69    19510.69    0.00       0          -1
2025-01-15 09:20:00  19507.95    19509.78    19510.35    -0.57      0          -1
2025-01-15 09:25:00  19520.95    19513.50    19511.67    1.83       1           1  ← Bullish
2025-01-15 09:30:00  19551.04    19526.01    19516.59    9.42       0           1
2025-01-15 09:35:00  19546.38    19532.80    19520.32    12.49      0           1
...
2025-01-15 10:25:00  19508.94    19545.69    19549.25    -3.56      -1         -1  ← Bearish
```

---

## How to Use

### Loading Data with EMA

```python
import pandas as pd

df = pd.read_csv('data/processed/nifty_features_5min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get bullish crossover signals
bullish_signals = df[df['ema_crossover'] == 1]

# Get bearish crossover signals
bearish_signals = df[df['ema_crossover'] == -1]

# Filter uptrend periods
uptrend = df[df['ema_trend'] == 1]
```

### Trading Logic Example

```python
# Simple crossover strategy
for i, row in df.iterrows():
    if row['ema_crossover'] == 1:
        print(f"BUY signal at {row['timestamp']}, price: {row['spot_close']}")
    elif row['ema_crossover'] == -1:
        print(f"SELL signal at {row['timestamp']}, price: {row['spot_close']}")
```

---

## Why EMA for Trading?

### Advantages

1. **Responsive**: Reacts faster to price changes than SMA
2. **Trend Following**: Identifies trend direction clearly
3. **Signal Generation**: Crossovers provide clear entry/exit points
4. **Noise Filtering**: Smooths out random price fluctuations

### Limitations

1. **Lagging**: Still a lagging indicator (based on past prices)
2. **Whipsaws**: Can generate false signals in choppy markets
3. **No Prediction**: Doesn't predict future prices, only identifies trends

---

## Output File

| File | Location | New Columns |
|------|----------|-------------|
| `nifty_features_5min.csv` | `data/processed/` | 6 EMA columns added |

---

## Next Steps

The EMA indicators will be used in:
- **Task 2.2**: Additional technical indicators (RSI, MACD, etc.)
- **Task 4**: Trading strategy implementation
- **Task 5**: Machine learning features

---

## Glossary

| Term | Definition |
|------|------------|
| **EMA** | Exponential Moving Average - weighted average favoring recent prices |
| **Period** | Number of data points used in calculation |
| **Alpha/k** | Smoothing factor determining weight of recent prices |
| **Crossover** | When one moving average crosses another |
| **Whipsaw** | False signal caused by rapid trend reversals |
