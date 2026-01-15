# Task 2.4: Final Feature Set

## Overview

This document describes the final consolidated feature set containing all EMA indicators, Options Greeks, IV features, and derived features for the quantitative trading system.

---

## Deliverable

| File | Location | Records | Features |
|------|----------|---------|----------|
| `nifty_features_5min.csv` | `data/processed/` | 19,912 | 115 columns |

---

## Feature Categories Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Time Features** | 6 | timestamp, date, time, day_of_week, hour, minute |
| **Spot Data** | 9 | OHLCV + returns + volatility |
| **Futures Data** | 14 | OHLCV + OI + basis + returns |
| **EMA Indicators** | 6 | Fast/slow EMA, crossovers, trend |
| **Options Data** | 59 | CE/PE for ATM±2 strikes |
| **Greeks** | 10 | Delta, Gamma, Theta, Vega, Rho (CE/PE) |
| **IV Features** | 3 | Average IV, IV spread, IV percentile |
| **PCR Features** | 2 | OI-based and Volume-based PCR |
| **Derived Features** | 6 | GEX, delta neutral ratio, momentum |

---

## Complete Column Reference

### Time Features (6)
```
timestamp, date, time, day_of_week, hour, minute
```

### Spot Data (9)
```
spot_open, spot_high, spot_low, spot_close, spot_volume
spot_return_1, spot_return_5, spot_log_return, spot_volatility_1h
```

### Futures Data (14)
```
fut_open, fut_high, fut_low, fut_close, fut_volume, fut_oi
fut_contract_month, fut_dte, fut_close_adjusted
fut_basis, fut_basis_pct
fut_return_1, fut_return_5, fut_log_return
```

### EMA Indicators (6)
```
ema_fast (5-period), ema_slow (15-period)
ema_diff, ema_diff_pct
ema_crossover, ema_trend
```

### Options Data (59)
For each strike (ATM, ATM±1, ATM±2) and type (CE, PE):
```
opt_{ce/pe}_{atm/atm_p1/atm_p2/atm_m1/atm_m2}_{strike/ltp/iv/oi/vol}
opt_expiry, opt_dte
opt_atm_straddle, opt_atm_iv_avg, opt_atm_pcr_oi, opt_atm_pcr_vol
opt_total_ce_oi, opt_total_pe_oi, opt_total_pcr_oi
```

### Greeks (10)
```
greeks_ce_atm_delta, greeks_ce_atm_gamma, greeks_ce_atm_theta
greeks_ce_atm_vega, greeks_ce_atm_rho
greeks_pe_atm_delta, greeks_pe_atm_gamma, greeks_pe_atm_theta
greeks_pe_atm_vega, greeks_pe_atm_rho
```

### IV Features (3)
```
avg_iv, iv_spread, iv_percentile
```

### PCR Features (2)
```
pcr_oi, pcr_volume
```

### Derived Features (6)
```
futures_basis_pct, delta_neutral_ratio
gex_ce, gex_pe, gex_net
price_momentum
```

---

## Data Quality

| Metric | Value |
|--------|-------|
| Total Records | 19,912 |
| Date Range | 2025-01-15 to 2026-01-15 |
| Rows with Options Data | 1,565 (hourly samples) |
| Rows with Greeks | 1,565 |
| Missing Values | 60.96% (due to hourly options sampling) |

**Note**: The high missing value percentage is expected because options data is sampled hourly while spot/futures data is at 5-minute intervals. This preserves the granularity of price data while managing data size.

---

## Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Task 1.3: Merged Data                                       │
│  └── nifty_merged_5min.csv (81 columns)                     │
│           │                                                  │
│           ▼                                                  │
│  Task 2.1: EMA Indicators (+6 columns)                      │
│  └── ema_fast, ema_slow, ema_diff, ema_crossover, etc.     │
│           │                                                  │
│           ▼                                                  │
│  Task 2.2: Options Greeks (+10 columns)                     │
│  └── Delta, Gamma, Theta, Vega, Rho for CE/PE              │
│           │                                                  │
│           ▼                                                  │
│  Task 2.3: Derived Features (+18 columns)                   │
│  └── IV, PCR, Returns, GEX, etc.                           │
│           │                                                  │
│           ▼                                                  │
│  Task 2.4: Final Feature Set (115 columns)                  │
│  └── nifty_features_5min.csv                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Use

### Loading the Data

```python
import pandas as pd

df = pd.read_csv('data/processed/nifty_features_5min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Records: {len(df)}")
print(f"Features: {len(df.columns)}")
```

### Filtering by Feature Category

```python
# Get only EMA features
ema_cols = [c for c in df.columns if c.startswith('ema_')]

# Get only Greeks
greeks_cols = [c for c in df.columns if c.startswith('greeks_')]

# Get rows with complete options data
df_with_options = df.dropna(subset=['opt_ce_atm_ltp'])
```

### Example Analysis

```python
# Trading signals based on EMA crossover
buy_signals = df[df['ema_crossover'] == 1]
sell_signals = df[df['ema_crossover'] == -1]

# High IV environment
high_iv = df[df['avg_iv'] > 18]

# Bearish sentiment
bearish = df[df['pcr_oi'] > 1.2]

# Negative GEX (expect volatility)
volatile = df[df['gex_net'] < 0]
```

---

## Validation Results

All required features validated:

| Category | Status |
|----------|--------|
| EMA Features | ✓ Present |
| Greeks (CE) | ✓ Present |
| Greeks (PE) | ✓ Present |
| IV Features | ✓ Present |
| PCR Features | ✓ Present |
| Returns | ✓ Present |
| Derived Features | ✓ Present |

---

## Next Steps

The final feature set is ready for:
- **Part 3**: Regime Detection
- **Part 4**: Trading Strategy Implementation
- **Part 5**: Machine Learning Models
- **Part 6**: Backtesting and Analysis

---

## Part 2 Summary

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | EMA Indicators | ema_fast, ema_slow, crossovers |
| 2.2 | Options Greeks | Delta, Gamma, Theta, Vega, Rho |
| 2.3 | Derived Features | IV, PCR, Returns, GEX |
| 2.4 | Final Feature Set | `nifty_features_5min.csv` |

**Total Features Added**: 34 new columns
**Final Dataset**: 115 columns, 19,912 records
