# Task 1.3: Data Merging

## Overview

This document explains how we merged the three cleaned datasets (spot, futures, options) into a single unified dataset for analysis and trading strategy development.

---

## Why Merge Data?

Think of it like assembling a puzzle:
- **Individual pieces** (separate datasets) are useful but incomplete
- **Complete puzzle** (merged dataset) gives you the full picture

For trading, we need to see:
- What's the spot price? (spot data)
- What's the futures price and basis? (futures data)
- What are options pricing and IV? (options data)

**All at the same moment in time** - that's why we merge on timestamp.

---

## Merging Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Merging Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SPOT DATA (19,912 rows)                                    │
│  ├── timestamp                                               │
│  ├── spot_open, spot_high, spot_low, spot_close             │
│  └── spot_volume                                             │
│           │                                                  │
│           ▼ INNER JOIN on timestamp                          │
│                                                              │
│  FUTURES DATA (19,912 rows)                                 │
│  ├── timestamp                                               │
│  ├── fut_open, fut_high, fut_low, fut_close                 │
│  ├── fut_volume, fut_oi                                      │
│  ├── fut_basis, fut_basis_pct                               │
│  └── fut_contract_month, fut_dte                            │
│           │                                                  │
│           ▼ LEFT JOIN on timestamp                           │
│                                                              │
│  OPTIONS DATA (pivoted, 1,565 unique timestamps)            │
│  ├── timestamp                                               │
│  ├── opt_ce_atm_*, opt_pe_atm_* (ATM options)               │
│  ├── opt_ce_atm_p1_*, opt_pe_atm_p1_* (ATM+1)              │
│  ├── opt_ce_atm_m1_*, opt_pe_atm_m1_* (ATM-1)              │
│  └── Derived: straddle, PCR, total OI                       │
│           │                                                  │
│           ▼                                                  │
│                                                              │
│  MERGED DATA (19,912 rows, 81 columns)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Join Types Explained

### Inner Join (Spot + Futures)
- Keeps only rows where timestamp exists in **both** datasets
- Since spot and futures have identical timestamps, all 19,912 rows are kept

### Left Join (with Options)
- Keeps **all** rows from the left side (spot+futures)
- Adds options data where available
- Options data is hourly (sampled), so many rows have `NaN` for options columns
- This is intentional - we don't want to lose 5-minute granularity

---

## Column Naming Convention

We use prefixes to identify data source:

| Prefix | Source | Example |
|--------|--------|---------|
| `spot_` | NIFTY 50 Index | `spot_close`, `spot_volume` |
| `fut_` | NIFTY Futures | `fut_close`, `fut_oi`, `fut_basis` |
| `opt_` | NIFTY Options | `opt_ce_atm_ltp`, `opt_pe_atm_iv` |

### Options Column Naming

Options columns follow this pattern:
```
opt_{type}_{strike}_{metric}

Where:
- type: ce (Call) or pe (Put)
- strike: atm, atm_p1, atm_p2, atm_m1, atm_m2
- metric: ltp, iv, oi, vol, strike
```

**Examples:**
- `opt_ce_atm_ltp` - Call option ATM Last Traded Price
- `opt_pe_atm_m1_iv` - Put option ATM-1 Implied Volatility
- `opt_ce_atm_p2_oi` - Call option ATM+2 Open Interest

---

## Derived Metrics

We calculate additional useful metrics during merging:

### Futures Metrics

| Column | Formula | Description |
|--------|---------|-------------|
| `fut_basis` | `fut_close - spot_close` | Absolute basis in points |
| `fut_basis_pct` | `(fut_basis / spot_close) × 100` | Basis as percentage |

### Options Metrics

| Column | Formula | Description |
|--------|---------|-------------|
| `opt_atm_straddle` | `CE_ATM_LTP + PE_ATM_LTP` | ATM straddle price |
| `opt_atm_iv_avg` | `(CE_ATM_IV + PE_ATM_IV) / 2` | Average ATM IV |
| `opt_atm_pcr_oi` | `PE_ATM_OI / CE_ATM_OI` | Put-Call Ratio (OI) |
| `opt_atm_pcr_vol` | `PE_ATM_VOL / CE_ATM_VOL` | Put-Call Ratio (Volume) |
| `opt_total_ce_oi` | Sum of all CE OI | Total Call OI |
| `opt_total_pe_oi` | Sum of all PE OI | Total Put OI |
| `opt_total_pcr_oi` | `Total_PE_OI / Total_CE_OI` | Overall PCR |

### Time Components

| Column | Description |
|--------|-------------|
| `date` | Date only (for grouping) |
| `time` | Time only |
| `day_of_week` | 0=Monday, 4=Friday |
| `hour` | Hour of day (9-15) |
| `minute` | Minute (0, 5, 10, ..., 55) |

---

## Merged Dataset Summary

### Structure

| Category | Columns | Description |
|----------|---------|-------------|
| Spot | 5 | OHLCV data |
| Futures | 11 | OHLCV, OI, basis, contract info |
| Options CE | 26 | Call options for 5 strikes |
| Options PE | 26 | Put options for 5 strikes |
| Options Derived | 7 | Straddle, PCR, etc. |
| Time | 6 | timestamp, date, time, etc. |
| **Total** | **81** | |

### Records

- **Total rows**: 19,912
- **With options data**: ~1,565 (hourly samples)
- **Date range**: 2025-01-15 to 2026-01-15
- **Trading days**: 262

---

## Sample Data

```
timestamp            spot_close  fut_close  fut_basis  opt_atm_straddle
2025-01-15 09:15:00  19510.69    19551.98   41.29      510.29
2025-01-15 09:20:00  19507.95    19551.88   43.93      NaN
2025-01-15 09:25:00  19520.95    19560.55   39.60      NaN
...
2025-01-15 10:15:00  19545.32    19585.10   39.78      498.45
```

Note: Options data is `NaN` for non-hourly timestamps (by design).

---

## How to Use the Merged Data

### Loading the Data

```python
import pandas as pd

df = pd.read_csv('data/processed/nifty_merged_5min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
```

### Common Operations

```python
# Get only rows with options data
df_with_options = df.dropna(subset=['opt_atm_straddle'])

# Filter by date
df_day = df[df['date'] == '2025-06-15']

# Get morning session only
df_morning = df[df['hour'] < 12]

# Calculate daily VWAP
df['vwap'] = (df['spot_close'] * df['spot_volume']).cumsum() / df['spot_volume'].cumsum()
```

---

## Output File

| File | Location | Size |
|------|----------|------|
| `nifty_merged_5min.csv` | `data/processed/` | ~19,912 rows × 81 columns |

---

## Next Steps

The merged dataset is ready for:
- **Part 2**: Feature Engineering (technical indicators)
- **Part 3**: Regime Detection
- **Part 4**: Strategy Implementation
- **Part 5**: Machine Learning

---

## Glossary

| Term | Definition |
|------|------------|
| **Inner Join** | Keeps only matching rows from both tables |
| **Left Join** | Keeps all rows from left table, adds matching from right |
| **Basis** | Difference between futures and spot price |
| **Straddle** | Combined price of ATM Call + ATM Put |
| **PCR** | Put-Call Ratio - measure of market sentiment |
| **ATM±n** | Strikes above (+) or below (-) At-The-Money |
