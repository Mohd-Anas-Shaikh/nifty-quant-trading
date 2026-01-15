# Task 1.2: Data Cleaning

## Overview

This document explains the data cleaning process for our quantitative trading system. Data cleaning is a critical step that ensures our trading algorithms work with accurate, consistent, and reliable data.

---

## Why Data Cleaning Matters

Think of data cleaning like preparing ingredients before cooking:
- **Raw ingredients** (raw data) may have dirt, bruises, or inconsistencies
- **Cleaned ingredients** (cleaned data) are ready to use and will produce better results
- **Bad ingredients** lead to bad dishes; **bad data** leads to bad trading decisions

In quantitative trading, even small data errors can lead to:
- False trading signals
- Incorrect risk calculations
- Significant financial losses

---

## What We Clean

### 1. Missing Values

**What are they?**
Missing values are gaps in our data where information should exist but doesn't. Like a book with pages torn out.

**Why do they occur?**
- Network issues during data transmission
- Exchange system downtime
- Data provider errors
- Trading halts

**How we handle them:**

```
Strategy: Forward Fill → Backward Fill → Interpolation

Example:
Time     Price (Raw)    Price (Cleaned)
09:15    19500          19500
09:20    [MISSING]  →   19500  (forward fill)
09:25    19520          19520
09:30    [MISSING]  →   19520  (forward fill)
09:35    19540          19540
```

**Why this approach?**
- **Forward fill**: In markets, the last known price is the best estimate for a missing value
- **Backward fill**: Used only for missing values at the start of data
- **Linear interpolation**: For rare cases with multiple consecutive missing values

---

### 2. Outliers

**What are they?**
Outliers are data points that are significantly different from other observations. Like finding a $1 million price tag on a $10 item.

**Why do they occur?**
- Data entry errors
- System glitches ("fat finger" trades)
- Flash crashes
- Incorrect decimal placement

**How we detect them:**

We use the **IQR (Interquartile Range) Method**:

```
┌─────────────────────────────────────────────────────────────┐
│                    IQR Method Explained                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data sorted: [10, 12, 14, 15, 16, 18, 20, 100]             │
│                                                              │
│  Q1 (25th percentile) = 13                                  │
│  Q3 (75th percentile) = 19                                  │
│  IQR = Q3 - Q1 = 6                                          │
│                                                              │
│  Lower Bound = Q1 - 1.5 × IQR = 13 - 9 = 4                  │
│  Upper Bound = Q3 + 1.5 × IQR = 19 + 9 = 28                 │
│                                                              │
│  Outlier: 100 (exceeds upper bound of 28)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How we correct them:**
- Replace outliers with the **rolling median** (average of nearby values)
- This preserves the general trend while removing extreme spikes

---

### 3. Timestamp Alignment

**What is it?**
Ensuring all three datasets (spot, futures, options) have matching timestamps so we can compare them accurately.

**Why is it important?**
Imagine comparing today's apple price with yesterday's orange price - it doesn't make sense. We need to compare data from the same moment in time.

**Our approach:**

```
┌─────────────────────────────────────────────────────────────┐
│                  Timestamp Alignment                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Spot Data:      [09:15, 09:20, 09:25, 09:30, 09:35]        │
│  Futures Data:   [09:15, 09:20, 09:25, 09:30, 09:35]        │
│  Options Data:   [09:15,       09:25,       09:35]          │
│                                                              │
│  Common (Spot+Futures): [09:15, 09:20, 09:25, 09:30, 09:35] │
│  Common (All):          [09:15, 09:25, 09:35]               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Result:**
- Spot and Futures: 19,912 aligned records
- Options: 15,650 records (sampled hourly for efficiency)

---

### 4. Futures Contract Rollover

**What is rollover?**
NIFTY futures contracts expire on the last Thursday of each month. Before expiry, traders "roll over" to the next month's contract.

**The problem:**
When a new contract starts, its price may be different from the old contract, creating artificial price jumps.

```
Example without adjustment:
Date        Contract    Price
Jan 29      JAN-FUT     19500
Jan 30      JAN-FUT     19520  (expiry day)
Jan 31      FEB-FUT     19580  ← Artificial jump!

The 60-point jump isn't a real market move - it's just the 
difference between contract prices (called "basis").
```

**Our solution: Ratio Adjustment Method**

```
┌─────────────────────────────────────────────────────────────┐
│                  Ratio Adjustment Method                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  At rollover:                                                │
│    Old contract price: 19520                                 │
│    New contract price: 19580                                 │
│    Ratio = 19580 / 19520 = 1.00307                          │
│                                                              │
│  Adjust all historical prices:                               │
│    Adjusted Price = Original Price × Ratio                   │
│                                                              │
│  Result: Smooth, continuous price series                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why ratio method?**
- Preserves **percentage returns** (what traders actually care about)
- Creates a **continuous series** for backtesting
- Industry standard approach

**Rollovers detected:** 11 (one per month over the year)

---

### 5. Dynamic ATM Strike Calculation

**What is ATM (At-The-Money)?**
An option is ATM when its strike price equals the current spot price.

**The challenge:**
The spot price changes every 5 minutes, so the ATM strike also changes throughout the day.

```
Example:
Time     Spot Price    ATM Strike (rounded to nearest 50)
09:15    19510         19500
10:30    19580         19600  ← ATM shifted!
14:00    19620         19600
15:00    19490         19500  ← ATM shifted back!
```

**Our approach:**

```python
ATM Strike = Round(Spot Price / 50) × 50

Example:
  Spot = 19527
  ATM = Round(19527 / 50) × 50
      = Round(390.54) × 50
      = 391 × 50
      = 19550
```

**Why dynamic calculation?**
- Ensures options analysis uses the **correct ATM reference**
- Critical for strategies like straddles and strangles
- Affects Greeks calculations and IV analysis

---

## Cleaning Results Summary

| Dataset | Original | Cleaned | Missing Fixed | Outliers Fixed |
|---------|----------|---------|---------------|----------------|
| Spot | 19,912 | 19,912 | 0 | 0 |
| Futures | 19,912 | 19,912 | 0 | 1 |
| Options | 15,650 | 15,650 | 0 | 0 |

### Key Statistics After Cleaning

**Spot Data:**
- Price Range: ₹19,188.51 to ₹24,910.84
- Mean Price: ₹21,118.03

**Futures Data:**
- 11 contract rollovers handled
- Mean Basis: ₹39.53 (futures premium over spot)
- Continuous adjusted price series created

**Options Data:**
- IV Range: 10.06% to 25.85%
- Mean IV: 17.41%
- ATM recalculated for 1,565 timestamps

**Data Quality Score: 100%**

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `nifty_spot_5min_cleaned.csv` | `data/processed/` | Cleaned spot OHLCV |
| `nifty_futures_5min_cleaned.csv` | `data/processed/` | Cleaned futures with adjusted prices |
| `nifty_options_5min_cleaned.csv` | `data/processed/` | Cleaned options with dynamic ATM |
| `data_cleaning_report.txt` | `data/processed/` | Detailed cleaning statistics |

---

## Code Structure

```
src/data/
├── data_cleaner.py          # Main cleaning module
│   ├── DataCleaningReport   # Report generation class
│   └── DataCleaner          # Main cleaning class
│       ├── detect_missing_values()
│       ├── impute_missing_values()
│       ├── detect_outliers_iqr()
│       ├── correct_outliers()
│       ├── clean_spot_data()
│       ├── clean_futures_data()
│       ├── clean_options_data()
│       ├── _handle_futures_rollover()
│       ├── _calculate_dynamic_atm()
│       └── align_timestamps()
```

---

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run data cleaning
python run_data_cleaning.py
```

---

## Quality Assurance Checks

The cleaning process validates:

1. **OHLC Consistency**: High ≥ max(Open, Close), Low ≤ min(Open, Close)
2. **Positive Values**: All prices, volumes, and OI are non-negative
3. **IV Bounds**: Implied volatility between 5% and 100%
4. **Timestamp Order**: Data is chronologically sorted
5. **No Duplicates**: Each timestamp appears only once per instrument

---

## Next Steps

The cleaned data is now ready for:
- **Task 2**: Feature Engineering (technical indicators, derived metrics)
- **Task 3**: Regime Detection (market state identification)
- **Task 4**: Strategy Implementation (trading rules)

---

## Glossary

| Term | Definition |
|------|------------|
| **OHLCV** | Open, High, Low, Close, Volume - standard price data format |
| **IQR** | Interquartile Range - measure of statistical dispersion |
| **ATM** | At-The-Money - option strike equal to spot price |
| **Basis** | Difference between futures and spot price |
| **Rollover** | Switching from expiring contract to next month's contract |
| **IV** | Implied Volatility - market's expectation of future price movement |
| **Forward Fill** | Using previous value to fill missing data |
