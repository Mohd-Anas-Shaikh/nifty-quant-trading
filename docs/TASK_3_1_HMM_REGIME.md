# Task 3.1: HMM Regime Detection

## Overview

This document explains the Hidden Markov Model (HMM) implementation for market regime detection. The model classifies the market into three regimes based on options-derived features.

---

## What is a Hidden Markov Model?

Think of the market like weather:
- The **actual weather** (sunny, cloudy, rainy) is the **hidden state** - we can't directly observe it
- What we **can observe** are things like temperature, humidity, wind - these are **emissions**
- The weather tends to **persist** - sunny days often follow sunny days
- But it can **transition** - sunny can become cloudy

HMM works the same way for markets:
- **Hidden states** = Market regimes (Uptrend, Downtrend, Sideways)
- **Observations** = Features we can measure (IV, PCR, Greeks, etc.)
- **Persistence** = Markets tend to stay in a regime for a while
- **Transitions** = But regimes do change

```
┌─────────────────────────────────────────────────────────────┐
│                  HIDDEN MARKOV MODEL                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HIDDEN STATES (what we want to know):                      │
│                                                              │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│     │ Uptrend  │ ←──→ │ Sideways │ ←──→ │Downtrend │       │
│     │   (+1)   │      │   (0)    │      │   (-1)   │       │
│     └────┬─────┘      └────┬─────┘      └────┬─────┘       │
│          │                 │                 │              │
│          ▼                 ▼                 ▼              │
│                                                              │
│  OBSERVATIONS (what we can measure):                        │
│     IV, PCR, Delta, Gamma, Vega, Basis, Returns             │
│                                                              │
│  The model learns:                                           │
│  1. Transition probabilities between states                 │
│  2. What observations to expect in each state               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Regime Definitions

| Regime | Value | Description | Characteristics |
|--------|-------|-------------|-----------------|
| **Uptrend** | +1 | Bullish market | Positive returns, rising prices |
| **Sideways** | 0 | Range-bound | Low volatility, mean-reverting |
| **Downtrend** | -1 | Bearish market | Negative returns, falling prices |

---

## Input Features

We use **options-based features** because they reflect market expectations:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `avg_iv` | Average Implied Volatility | High IV = uncertainty, often in downtrends |
| `iv_spread` | Call IV - Put IV | Skew indicates directional bias |
| `pcr_oi` | Put-Call Ratio (OI) | High PCR = bearish positioning |
| `greeks_ce_atm_delta` | ATM Call Delta | Sensitivity to price moves |
| `greeks_ce_atm_gamma` | ATM Call Gamma | Rate of delta change |
| `greeks_ce_atm_vega` | ATM Call Vega | Sensitivity to IV changes |
| `futures_basis_pct` | Futures Basis (%) | Cost of carry, sentiment |
| `spot_return_1` | Spot Returns | Direct price momentum |

---

## Training Process

### Data Split
- **Training**: First 70% of data (1,094 samples)
- **Testing**: Remaining 30% (470 samples)

### Algorithm: Expectation-Maximization (EM)

The HMM uses the EM algorithm to learn:

1. **E-step**: Given current parameters, estimate which regime each observation belongs to
2. **M-step**: Given regime assignments, update parameters to maximize likelihood
3. **Repeat** until convergence

### Regime Mapping

After training, we map HMM states to meaningful regimes based on **average returns**:
- State with **highest** average return → Uptrend (+1)
- State with **lowest** average return → Downtrend (-1)
- State in **middle** → Sideways (0)

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Training Log-Likelihood | -4,791.74 |
| Test Log-Likelihood | -3,213.84 |
| Training Samples | 1,094 |
| Test Samples | 470 |

### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| **Uptrend (+1)** | 396 | 25.32% |
| **Sideways (0)** | 865 | 55.31% |
| **Downtrend (-1)** | 303 | 19.37% |

**Interpretation**: The market spends most time in sideways regime (55%), with roughly equal time in uptrend (25%) and downtrend (19%).

### Transition Matrix

The transition matrix shows the probability of moving from one regime to another:

```
                    To:
From:        Uptrend   Downtrend   Sideways
Uptrend      97.2%     2.8%        0.0%
Downtrend    0.0%      96.1%       3.9%
Sideways     1.3%      0.0%        98.7%
```

**Key Insights**:
- **High persistence**: Each regime has >96% probability of staying in the same regime
- **Uptrend → Downtrend**: 2.8% chance (direct reversal)
- **Downtrend → Sideways**: 3.9% chance (recovery through consolidation)
- **Sideways → Uptrend**: 1.3% chance (breakout)

---

## Features Added to Dataset

| Column | Description | Values |
|--------|-------------|--------|
| `regime` | Predicted regime | -1, 0, +1 |
| `regime_label` | Text label | Downtrend, Sideways, Uptrend |
| `regime_prob_down` | Probability of downtrend | 0.0 to 1.0 |
| `regime_prob_side` | Probability of sideways | 0.0 to 1.0 |
| `regime_prob_up` | Probability of uptrend | 0.0 to 1.0 |

---

## How to Use Regimes

### Basic Usage

```python
import pandas as pd

df = pd.read_csv('data/processed/nifty_features_5min.csv')

# Filter by regime
uptrend = df[df['regime'] == 1]
downtrend = df[df['regime'] == -1]
sideways = df[df['regime'] == 0]

# Trade only in uptrend
if current_regime == 1:
    execute_long_strategy()
```

### Regime-Based Strategy

```python
# Different strategies for different regimes
def get_strategy(regime):
    if regime == 1:  # Uptrend
        return 'trend_following_long'
    elif regime == -1:  # Downtrend
        return 'trend_following_short'
    else:  # Sideways
        return 'mean_reversion'
```

### Using Probabilities

```python
# Only trade when regime is confident (>80% probability)
confident_uptrend = df[df['regime_prob_up'] > 0.8]
confident_downtrend = df[df['regime_prob_down'] > 0.8]
```

---

## Model Files

| File | Location | Description |
|------|----------|-------------|
| `nifty_features_5min.csv` | `data/processed/` | Dataset with regime columns |
| `hmm_regime_model.pkl` | `data/processed/` | Saved HMM model |

### Loading Saved Model

```python
from src.regime.hmm_regime import HMMRegimeDetector

detector = HMMRegimeDetector()
detector.load_model('data/processed/hmm_regime_model.pkl')

# Predict on new data
regimes, indices = detector.predict(new_df)
```

---

## Why HMM for Regime Detection?

### Advantages

1. **Captures persistence**: Markets don't randomly flip between regimes
2. **Probabilistic**: Gives confidence levels, not just hard classifications
3. **Unsupervised**: Learns patterns without labeled data
4. **Interpretable**: Transition matrix shows regime dynamics

### Limitations

1. **Fixed number of states**: Must specify 3 regimes upfront
2. **Gaussian assumption**: Assumes features are normally distributed
3. **Sequential data**: Requires time-ordered observations
4. **Lookback bias**: Uses future data in training (mitigated by train/test split)

---

## Glossary

| Term | Definition |
|------|------------|
| **HMM** | Hidden Markov Model - probabilistic model for sequential data |
| **Hidden State** | Unobservable regime (what we're trying to infer) |
| **Emission** | Observable features that depend on hidden state |
| **Transition Probability** | Likelihood of switching from one regime to another |
| **EM Algorithm** | Expectation-Maximization - method to train HMM |
| **Log-Likelihood** | Measure of how well model fits data (higher is better) |
