# Task 2.3: Derived Features

## Overview

This document explains the derived features created by combining multiple data points. These features provide meaningful signals for trading strategies and risk management.

---

## Features Created

### 1. Average IV

**Formula**: `Average IV = (Call IV + Put IV) / 2`

**What it tells you**: The market's overall expectation of future volatility, smoothing out any skew between calls and puts.

```
┌─────────────────────────────────────────────────────────────┐
│                      AVERAGE IV                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Call IV = 16%    Put IV = 14%                              │
│  Average IV = (16 + 14) / 2 = 15%                           │
│                                                              │
│  Interpretation:                                             │
│  • High Avg IV (>20%): Market expects big moves             │
│  • Low Avg IV (<12%): Market expects calm conditions        │
│  • Rising IV: Uncertainty increasing                        │
│  • Falling IV: Uncertainty decreasing                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = 15.02%, Std = 1.04%

---

### 2. IV Spread

**Formula**: `IV Spread = Call IV - Put IV`

**What it tells you**: Market sentiment through options pricing.

```
┌─────────────────────────────────────────────────────────────┐
│                      IV SPREAD                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Positive Spread (Call IV > Put IV):                        │
│  • Calls are more expensive                                 │
│  • Bullish sentiment or upside hedging                      │
│                                                              │
│  Negative Spread (Put IV > Call IV):                        │
│  • Puts are more expensive                                  │
│  • Bearish sentiment or downside protection demand          │
│  • Common before uncertain events                           │
│                                                              │
│  Example:                                                    │
│  Call IV = 14%, Put IV = 16%                                │
│  IV Spread = 14 - 16 = -2%                                  │
│  → Puts are expensive, market is hedging downside           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = -0.07%, Range = [-6.69%, +7.45%]

---

### 3. PCR (OI-based)

**Formula**: `PCR (OI) = Total Put Open Interest / Total Call Open Interest`

**What it tells you**: The positioning of market participants based on outstanding contracts.

```
┌─────────────────────────────────────────────────────────────┐
│                    PCR (Open Interest)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PCR > 1.2: Heavy put positioning                           │
│  ├── Could be bearish bets                                  │
│  └── Or hedging (often contrarian bullish signal)           │
│                                                              │
│  PCR < 0.8: Heavy call positioning                          │
│  ├── Could be bullish bets                                  │
│  └── Or complacency (often contrarian bearish signal)       │
│                                                              │
│  PCR ≈ 1.0: Balanced market                                 │
│                                                              │
│  Note: PCR is often used as a CONTRARIAN indicator          │
│  Extreme readings often precede reversals                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = 1.01, Std = 0.13

---

### 4. PCR (Volume-based)

**Formula**: `PCR (Volume) = Total Put Volume / Total Call Volume`

**What it tells you**: Real-time trading activity sentiment.

```
┌─────────────────────────────────────────────────────────────┐
│                    PCR (Volume)                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  More responsive than OI-based PCR                          │
│  Shows what traders are doing RIGHT NOW                     │
│                                                              │
│  High Volume PCR: Active put buying                         │
│  Low Volume PCR: Active call buying                         │
│                                                              │
│  Use with OI PCR:                                            │
│  • Volume PCR rising + OI PCR stable = Short-term hedging   │
│  • Both rising = Building bearish positions                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = 1.03, Std = 0.23

---

### 5. Futures Basis (%)

**Formula**: `Futures Basis = (Futures Close - Spot Close) / Spot Close × 100`

**What it tells you**: The premium or discount of futures relative to spot.

```
┌─────────────────────────────────────────────────────────────┐
│                   FUTURES BASIS (%)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Normal Market (Contango):                                   │
│  Futures > Spot → Positive Basis                            │
│  • Reflects cost of carry (interest - dividends)            │
│  • Typically 0.1% to 0.3% for monthly contracts             │
│                                                              │
│  Inverted Market (Backwardation):                           │
│  Futures < Spot → Negative Basis                            │
│  • Rare, indicates strong selling pressure                  │
│  • Or high dividend expectations                            │
│                                                              │
│  Basis Convergence:                                          │
│  • Basis → 0 as expiry approaches                           │
│  • At expiry: Futures = Spot                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = 0.19%

---

### 6. Returns

**Formulas**:
- `Spot Return (1-period) = (Price_t - Price_{t-1}) / Price_{t-1} × 100`
- `Spot Return (5-period) = (Price_t - Price_{t-5}) / Price_{t-5} × 100`
- `Log Return = ln(Price_t / Price_{t-1}) × 100`

**What they tell you**: Price momentum and volatility.

```
┌─────────────────────────────────────────────────────────────┐
│                       RETURNS                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1-Period Return (5 minutes):                               │
│  • Short-term momentum                                       │
│  • Used for high-frequency signals                          │
│                                                              │
│  5-Period Return (25 minutes):                              │
│  • Medium-term momentum                                      │
│  • Smoother, less noisy                                     │
│                                                              │
│  Log Returns:                                                │
│  • Mathematically convenient (additive)                     │
│  • Better for statistical analysis                          │
│  • Similar to simple returns for small changes              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: 
- Spot Return (1-period): Mean = 0.001%, Std = 0.098%
- Futures returns also calculated

---

### 7. Delta Neutral Ratio

**Formula**: `Delta Neutral Ratio = |Call Delta| / |Put Delta|`

**What it tells you**: The hedge ratio needed for delta-neutral positions.

```
┌─────────────────────────────────────────────────────────────┐
│                 DELTA NEUTRAL RATIO                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ratio = 1.0: Perfect balance                               │
│  • 1 call hedges 1 put                                      │
│                                                              │
│  Ratio > 1.0: Call delta is larger                          │
│  • Need more puts to hedge calls                            │
│  • Example: Ratio = 1.2 → Need 1.2 puts per call           │
│                                                              │
│  Ratio < 1.0: Put delta is larger                           │
│  • Need more calls to hedge puts                            │
│                                                              │
│  Use case:                                                   │
│  • Constructing delta-neutral strategies                    │
│  • Straddles, strangles, iron condors                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Mean = 1.16

---

### 8. Gamma Exposure (GEX)

**Formula**: `GEX = Spot Price × Gamma × Open Interest`

**What it tells you**: Market maker positioning and potential market dynamics.

```
┌─────────────────────────────────────────────────────────────┐
│                   GAMMA EXPOSURE (GEX)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Positive GEX (Market makers long gamma):                   │
│  • MMs buy when price falls, sell when price rises          │
│  • DAMPENS volatility                                        │
│  • Price tends to mean-revert                               │
│                                                              │
│  Negative GEX (Market makers short gamma):                  │
│  • MMs sell when price falls, buy when price rises          │
│  • AMPLIFIES volatility                                      │
│  • Price tends to trend/accelerate                          │
│                                                              │
│  GEX Calculation:                                            │
│  • Call GEX = Spot × Call Gamma × Call OI (positive)        │
│  • Put GEX = -Spot × Put Gamma × Put OI (negative)          │
│  • Net GEX = Call GEX + Put GEX                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**: Net GEX Mean = 27,119.71

---

## Additional Features

| Feature | Description |
|---------|-------------|
| `spot_volatility_1h` | Rolling 1-hour volatility of returns |
| `price_momentum` | Price vs slow EMA (%) |
| `iv_percentile` | Current IV rank vs recent history |

---

## Complete Feature List

| Feature | Formula | Category |
|---------|---------|----------|
| `avg_iv` | (CE IV + PE IV) / 2 | IV |
| `iv_spread` | CE IV - PE IV | IV |
| `pcr_oi` | Total PE OI / Total CE OI | Sentiment |
| `pcr_volume` | Total PE Vol / Total CE Vol | Sentiment |
| `futures_basis_pct` | (Fut - Spot) / Spot × 100 | Basis |
| `spot_return_1` | 1-period % change | Returns |
| `spot_return_5` | 5-period % change | Returns |
| `spot_log_return` | Log return | Returns |
| `fut_return_1` | Futures 1-period return | Returns |
| `fut_return_5` | Futures 5-period return | Returns |
| `fut_log_return` | Futures log return | Returns |
| `delta_neutral_ratio` | \|CE Delta\| / \|PE Delta\| | Greeks |
| `gex_ce` | Spot × CE Gamma × CE OI | Greeks |
| `gex_pe` | -Spot × PE Gamma × PE OI | Greeks |
| `gex_net` | GEX CE + GEX PE | Greeks |
| `spot_volatility_1h` | Rolling std of returns | Volatility |
| `price_momentum` | (Spot - EMA) / EMA × 100 | Momentum |
| `iv_percentile` | IV rank (0-100) | IV |

---

## Summary Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Average IV | 15.02% | 1.04% | - | - |
| IV Spread | -0.07% | - | -6.69% | +7.45% |
| PCR (OI) | 1.01 | 0.13 | - | - |
| PCR (Volume) | 1.03 | 0.23 | - | - |
| Futures Basis | 0.19% | - | - | - |
| Spot Return | 0.001% | 0.098% | - | - |
| Delta Neutral Ratio | 1.16 | - | - | - |

---

## Output File

| File | Location | Total Columns |
|------|----------|---------------|
| `nifty_features_5min.csv` | `data/processed/` | 115 columns |

---

## How to Use

```python
import pandas as pd

df = pd.read_csv('data/processed/nifty_features_5min.csv')

# High IV environment
high_iv = df[df['avg_iv'] > 18]

# Bearish sentiment (high PCR)
bearish = df[df['pcr_oi'] > 1.2]

# Positive momentum
bullish_momentum = df[df['price_momentum'] > 0]

# Negative GEX (expect volatility)
volatile_regime = df[df['gex_net'] < 0]
```

---

## Glossary

| Term | Definition |
|------|------------|
| **IV** | Implied Volatility - market's expectation of future price movement |
| **PCR** | Put-Call Ratio - sentiment indicator |
| **Basis** | Difference between futures and spot price |
| **GEX** | Gamma Exposure - market maker positioning metric |
| **Contango** | Futures > Spot (normal market) |
| **Backwardation** | Futures < Spot (inverted market) |
