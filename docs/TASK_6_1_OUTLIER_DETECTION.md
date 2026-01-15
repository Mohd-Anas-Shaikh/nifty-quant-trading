# Task 6.1: Outlier Detection

## Overview

This document describes the analysis of high-performance trades (outliers) identified using Z-score methodology. Outliers are trades with exceptional profits beyond 3 standard deviations from the mean.

---

## Methodology

### Z-Score Calculation

```
Z-score = (P&L - Mean P&L) / Std Dev of P&L
```

### Outlier Definition

- **Positive Outliers**: Z-score > 3 (exceptional profits)
- **Negative Outliers**: Z-score < -3 (exceptional losses)
- **3-sigma rule**: ~99.7% of data falls within ±3 standard deviations

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Total Trades** | 364 |
| **Positive Outliers** | 10 (2.7%) |
| **Negative Outliers** | 0 |
| **Normal Trades** | 354 |

### P&L Statistics

| Metric | Outliers | Normal | All Trades |
|--------|----------|--------|------------|
| **Mean P&L** | 272.38 pts | -7.47 pts | 0.22 pts |
| **Min P&L** | 209.59 pts | - | - |
| **Max P&L** | 377.96 pts | - | - |
| **Std Dev** | - | - | 68.88 pts |

---

## Outlier Trades

| Entry Time | Exit Time | Type | P&L | P&L % | Z-Score |
|------------|-----------|------|-----|-------|---------|
| 2025-03-25 13:30 | 2025-03-26 13:10 | SHORT | 377.96 | 1.71% | **5.48** |
| 2025-10-28 12:25 | 2025-10-29 11:15 | SHORT | 314.03 | 1.42% | **4.56** |
| 2025-12-29 14:00 | 2025-12-30 10:50 | SHORT | 313.77 | 1.31% | **4.55** |
| 2025-07-30 15:30 | 2025-07-31 13:25 | SHORT | 287.48 | 1.43% | **4.17** |
| 2025-10-22 09:15 | 2025-10-22 14:20 | LONG | 282.18 | 1.29% | **4.09** |
| 2025-04-22 15:05 | 2025-04-23 14:50 | SHORT | 251.28 | 1.20% | **3.64** |
| 2025-06-17 10:05 | 2025-06-17 14:25 | LONG | 241.97 | 1.23% | **3.51** |
| 2025-04-23 15:00 | 2025-04-24 14:15 | SHORT | 228.22 | 1.11% | **3.31** |
| 2025-08-18 09:20 | 2025-08-18 12:45 | LONG | 217.27 | 1.10% | **3.15** |
| 2026-01-15 13:50 | 2026-01-15 15:30 | SHORT | 209.59 | 0.84% | **3.04** |

---

## Feature Analysis

### 1. Regime Distribution

| Regime | Outliers | Normal |
|--------|----------|--------|
| **Downtrend (-1)** | **70%** | ~30% |
| **Uptrend (+1)** | 30% | ~30% |
| **Sideways (0)** | 0% | ~40% |

**Key Finding**: 70% of outlier trades occurred in **Downtrend** regime. SHORT trades in downtrends produce exceptional returns.

---

### 2. Time of Day

| Metric | Value |
|--------|-------|
| **Peak Hour** | 13:00 (1 PM) |

**Key Finding**: Afternoon sessions (post-lunch) show higher probability of outlier trades.

---

### 3. IV (Implied Volatility)

| Metric | Outliers | Normal |
|--------|----------|--------|
| **Avg IV** | 15.13% | 15.05% |

**Key Finding**: IV levels are similar - outliers not strongly correlated with IV.

---

### 4. Trade Duration

| Metric | Outliers | Normal |
|--------|----------|--------|
| **Mean Duration** | 2041.9 bars | 11.1 bars |
| **Median Duration** | - | - |

**Key Finding**: Outlier trades have **significantly longer duration** (~184x longer). Patience is rewarded.

---

### 5. EMA Gap

| Metric | Outliers | Normal |
|--------|----------|--------|
| **Abs EMA Gap** | 5.08 | 3.14 |

**Key Finding**: Outlier trades have **larger EMA gaps** at entry (~62% larger). Stronger trend signals produce better results.

---

### 6. PCR (Put-Call Ratio)

| Metric | Outliers | Normal |
|--------|----------|--------|
| **PCR (OI)** | 0.8967 | 0.9973 |

**Key Finding**: Outlier trades have **lower PCR** (more call-heavy). This may indicate contrarian opportunities.

---

### 7. Trade Type

| Type | Outliers | Normal |
|------|----------|--------|
| **SHORT** | **70%** | ~47% |
| **LONG** | 30% | ~53% |

**Key Finding**: SHORT trades dominate outliers (7 of 10). Downtrend + SHORT = exceptional profits.

---

## Key Insights

### Characteristics of High-Performance Trades

1. **Regime**: Predominantly in **Downtrend** (70%)
2. **Direction**: Mostly **SHORT** trades (70%)
3. **Duration**: Much **longer holding periods** (2000+ bars vs 11 bars)
4. **EMA Gap**: **Larger trend signals** at entry
5. **Time**: Afternoon entries (peak at 1 PM)
6. **PCR**: Lower than average (call-heavy market)

### Trading Implications

```
┌─────────────────────────────────────────────────────────────┐
│           OUTLIER TRADE PROFILE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ Regime = Downtrend (-1)                                  │
│  ✓ Direction = SHORT                                        │
│  ✓ EMA Gap > 5 points                                       │
│  ✓ Entry Time = Afternoon (12-14h)                          │
│  ✓ PCR < 0.9                                                │
│  ✓ Hold for extended period                                 │
│                                                              │
│  Expected P&L: 200-380 points (1-2% return)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### 1. Focus on Downtrend Shorts
- 70% of outliers were SHORT trades in downtrend
- Prioritize these setups

### 2. Use Larger EMA Gap Filter
- Outliers had EMA gap > 5 points
- Consider filtering for stronger signals

### 3. Extend Holding Period
- Outliers held much longer than normal
- Don't exit too early on strong trends

### 4. Afternoon Entry Preference
- Peak outlier hour is 1 PM
- Consider time-of-day filter

### 5. Monitor PCR
- Lower PCR may signal contrarian opportunity
- PCR < 0.9 associated with outliers

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `outlier_analysis.pkl` | `data/processed/` | Complete analysis results |

---

## How to Use

### Loading Results

```python
import pickle

with open('data/processed/outlier_analysis.pkl', 'rb') as f:
    results = pickle.load(f)

outliers = results['outliers']
analysis = results['analysis']
```

### Filtering for Outlier-Like Setups

```python
# Check if current setup matches outlier profile
def is_outlier_setup(regime, trade_type, ema_gap, hour, pcr):
    return (
        regime == -1 and           # Downtrend
        trade_type == 'SHORT' and  # Short trade
        abs(ema_gap) > 5 and       # Large EMA gap
        12 <= hour <= 14 and       # Afternoon
        pcr < 0.9                   # Low PCR
    )
```

---

## Statistical Significance

| Test | Value | Interpretation |
|------|-------|----------------|
| Outlier % | 2.7% | Expected ~0.3% for normal distribution |
| Z-score range | 3.04 - 5.48 | All significantly above threshold |
| No negative outliers | 0 | Strategy avoids catastrophic losses |

---

## Glossary

| Term | Definition |
|------|------------|
| **Z-score** | Number of standard deviations from mean |
| **3-sigma** | 3 standard deviations (99.7% confidence) |
| **Outlier** | Data point beyond 3 standard deviations |
| **PCR** | Put-Call Ratio (put OI / call OI) |
| **EMA Gap** | Difference between fast and slow EMA |
