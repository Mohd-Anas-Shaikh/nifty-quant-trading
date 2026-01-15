# Task 2.2: Options Greeks and IV

## Overview

This document explains the Options Greeks calculated using the Black-Scholes model. Greeks measure how option prices change in response to various factors, helping traders understand and manage risk.

---

## What are Options Greeks?

Think of Greeks as the "sensitivity meters" of an option:
- **Delta**: How much does the option price move when the stock moves?
- **Gamma**: How fast does Delta change?
- **Theta**: How much value does the option lose each day?
- **Vega**: How sensitive is the option to volatility changes?
- **Rho**: How sensitive is the option to interest rate changes?

---

## The Black-Scholes Model

### What is it?

The Black-Scholes model is a mathematical formula that calculates the theoretical price of options. It was developed in 1973 and won a Nobel Prize in Economics.

### Assumptions

The model assumes:
1. Stock prices follow a random walk (log-normal distribution)
2. No dividends during the option's life
3. Constant volatility and interest rate
4. European-style options (exercise only at expiry)
5. No transaction costs

### Our Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk-free Rate | **6.5%** | Based on Indian government bond yields |
| Volatility | From IV | Implied volatility from market prices |
| Time | Days to Expiry / 365 | Converted to years |

---

## The Five Greeks Explained

### 1. Delta (Δ)

**What it measures**: How much the option price changes for a ₹1 change in the underlying.

```
┌─────────────────────────────────────────────────────────────┐
│                         DELTA                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CALL OPTIONS: Delta ranges from 0 to +1                    │
│  ─────────────────────────────────────────                  │
│  • Deep ITM Call: Delta ≈ 1.0 (moves like stock)           │
│  • ATM Call: Delta ≈ 0.5 (moves half as much)              │
│  • Deep OTM Call: Delta ≈ 0 (barely moves)                 │
│                                                              │
│  PUT OPTIONS: Delta ranges from -1 to 0                     │
│  ─────────────────────────────────────────                  │
│  • Deep ITM Put: Delta ≈ -1.0 (inverse of stock)           │
│  • ATM Put: Delta ≈ -0.5 (moves half, opposite)            │
│  • Deep OTM Put: Delta ≈ 0 (barely moves)                  │
│                                                              │
│  Example:                                                    │
│  If NIFTY moves up ₹100 and Call Delta = 0.5               │
│  Call option price increases by ₹100 × 0.5 = ₹50           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**:
- ATM Call Delta (mean): **0.537** (slightly ITM bias)
- ATM Put Delta (mean): **-0.463** (slightly OTM bias)

### 2. Gamma (Γ)

**What it measures**: How fast Delta changes when the underlying moves.

```
┌─────────────────────────────────────────────────────────────┐
│                         GAMMA                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Gamma is HIGHEST for ATM options                           │
│  Gamma is LOWEST for deep ITM/OTM options                   │
│                                                              │
│  Same for Calls and Puts at same strike                     │
│                                                              │
│         Gamma                                                │
│           │     ╱╲                                           │
│           │    ╱  ╲                                          │
│           │   ╱    ╲                                         │
│           │  ╱      ╲                                        │
│           │ ╱        ╲                                       │
│           └──────────────────                                │
│             OTM   ATM   ITM                                  │
│                                                              │
│  Why it matters:                                             │
│  High Gamma = Delta changes quickly = more risk/reward      │
│  ATM options near expiry have EXPLOSIVE Gamma               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**:
- ATM Call Gamma (mean): **0.000915**
- ATM Put Gamma (mean): **0.000911**

### 3. Theta (Θ)

**What it measures**: How much value the option loses per day (time decay).

```
┌─────────────────────────────────────────────────────────────┐
│                         THETA                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Theta is usually NEGATIVE (options lose value over time)   │
│                                                              │
│  Option Value                                                │
│       │╲                                                     │
│       │ ╲                                                    │
│       │  ╲                                                   │
│       │   ╲                                                  │
│       │    ╲__                                               │
│       │       ╲__                                            │
│       │          ╲__                                         │
│       └─────────────────                                     │
│       30 days    0 days                                      │
│                                                              │
│  Time decay ACCELERATES near expiry                         │
│  ATM options have highest Theta                             │
│                                                              │
│  Example:                                                    │
│  Theta = -5 means option loses ₹5 per day                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**:
- ATM Call Theta (mean): **-0.039** (loses ₹0.039 per day per ₹1 of spot)
- ATM Put Theta (mean): **-0.029** (loses ₹0.029 per day per ₹1 of spot)

### 4. Vega (ν)

**What it measures**: How much the option price changes for a 1% change in implied volatility.

```
┌─────────────────────────────────────────────────────────────┐
│                          VEGA                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Vega is always POSITIVE for both Calls and Puts            │
│  Higher volatility = Higher option prices                   │
│                                                              │
│  ATM options have highest Vega                              │
│  Longer-dated options have higher Vega                      │
│                                                              │
│  Example:                                                    │
│  Vega = 15 means:                                           │
│  If IV increases from 15% to 16%                            │
│  Option price increases by ₹15                              │
│                                                              │
│  Why it matters:                                             │
│  • Before events (earnings, elections): IV rises            │
│  • After events: IV falls ("volatility crush")              │
│  • Vega helps estimate these price changes                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**:
- ATM Call Vega (mean): **0.150** (per 1% IV change)
- ATM Put Vega (mean): **0.150** (per 1% IV change)

### 5. Rho (ρ)

**What it measures**: How much the option price changes for a 1% change in interest rates.

```
┌─────────────────────────────────────────────────────────────┐
│                          RHO                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CALL OPTIONS: Rho is POSITIVE                              │
│  Higher rates → Higher call prices                          │
│  (Cost of carry increases)                                  │
│                                                              │
│  PUT OPTIONS: Rho is NEGATIVE                               │
│  Higher rates → Lower put prices                            │
│  (Present value of strike decreases)                        │
│                                                              │
│  Rho is usually the LEAST important Greek                   │
│  Interest rates don't change frequently                     │
│                                                              │
│  More important for:                                         │
│  • Long-dated options (LEAPS)                               │
│  • Deep ITM options                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Our Results**:
- ATM Call Rho (mean): **0.042** (per 1% rate change)
- ATM Put Rho (mean): **-0.037** (per 1% rate change)

---

## Greeks Summary Table

| Greek | ATM Call (Mean) | ATM Put (Mean) | Interpretation |
|-------|-----------------|----------------|----------------|
| **Delta** | 0.537 | -0.463 | ~50% move correlation |
| **Gamma** | 0.000915 | 0.000911 | Delta changes slowly |
| **Theta** | -0.039 | -0.029 | Daily time decay |
| **Vega** | 0.150 | 0.150 | IV sensitivity |
| **Rho** | 0.042 | -0.037 | Rate sensitivity |

---

## Features Added to Dataset

| Column | Description |
|--------|-------------|
| `greeks_ce_atm_delta` | ATM Call Delta |
| `greeks_ce_atm_gamma` | ATM Call Gamma |
| `greeks_ce_atm_theta` | ATM Call Theta (daily) |
| `greeks_ce_atm_vega` | ATM Call Vega (per 1% IV) |
| `greeks_ce_atm_rho` | ATM Call Rho (per 1% rate) |
| `greeks_pe_atm_delta` | ATM Put Delta |
| `greeks_pe_atm_gamma` | ATM Put Gamma |
| `greeks_pe_atm_theta` | ATM Put Theta (daily) |
| `greeks_pe_atm_vega` | ATM Put Vega (per 1% IV) |
| `greeks_pe_atm_rho` | ATM Put Rho (per 1% rate) |

---

## Implementation Details

### Library Used

**py_vollib** - A Python library implementing Black-Scholes analytical formulas.

### Black-Scholes Formulas

```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Where:
  S = Spot price
  K = Strike price
  r = Risk-free rate (6.5%)
  σ = Implied volatility
  T = Time to expiry (years)
  N() = Cumulative normal distribution
```

### Greeks Formulas

| Greek | Call Formula | Put Formula |
|-------|--------------|-------------|
| Delta | N(d1) | N(d1) - 1 |
| Gamma | N'(d1) / (Sσ√T) | Same as Call |
| Theta | Complex (see code) | Complex (see code) |
| Vega | SN'(d1)√T | Same as Call |
| Rho | KTe^(-rT)N(d2) | -KTe^(-rT)N(-d2) |

---

## How to Use Greeks

### Risk Management

```python
# Calculate portfolio Delta
portfolio_delta = (num_calls * call_delta + num_puts * put_delta) * lot_size

# Delta-neutral hedging
# If portfolio_delta = 500, sell 500 shares of underlying to neutralize
```

### Trading Strategies

```python
# High Gamma near expiry = potential for big moves
if gamma > 0.002 and days_to_expiry < 3:
    print("Warning: High gamma risk!")

# Theta decay accelerates
if theta < -0.05:
    print("Significant daily decay - consider closing position")
```

---

## Output File

| File | Location | New Columns |
|------|----------|-------------|
| `nifty_features_5min.csv` | `data/processed/` | 10 Greeks columns added |

---

## Glossary

| Term | Definition |
|------|------------|
| **Black-Scholes** | Mathematical model for pricing European options |
| **Greeks** | Measures of option price sensitivity to various factors |
| **Delta** | Price sensitivity to underlying movement |
| **Gamma** | Rate of change of Delta |
| **Theta** | Time decay (daily value loss) |
| **Vega** | Sensitivity to volatility changes |
| **Rho** | Sensitivity to interest rate changes |
| **IV** | Implied Volatility - market's expectation of future volatility |
| **ATM** | At-The-Money - strike price equals spot price |
