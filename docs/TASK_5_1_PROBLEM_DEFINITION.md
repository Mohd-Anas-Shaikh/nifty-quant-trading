# Task 5.1: ML Problem Definition

## Overview

This document defines the machine learning problem for predicting trade profitability. We build a binary classifier to predict whether a trade signal will result in a profitable trade.

---

## Problem Definition

### Type
**Binary Classification**

### Objective
Predict if a trade signal will be profitable before entering the trade.

### Target Variable
```
Target = 1 if trade P&L > 0 (profitable)
Target = 0 if trade P&L ≤ 0 (unprofitable)
```

### Use Case
- Filter out potentially losing trades
- Improve win rate by only taking high-confidence signals
- Enhance the EMA regime strategy with ML predictions

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 355 |
| **Training Samples** | 248 (70%) |
| **Testing Samples** | 107 (30%) |
| **Total Features** | 86 |

### Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| **Profitable (1)** | 110 | 31.0% |
| **Unprofitable (0)** | 245 | 69.0% |

**Note**: Imbalanced dataset - more losing trades than winning trades (consistent with 30% win rate from backtest).

---

## Feature Categories

### 1. EMA Features (10)
Technical indicators from moving averages.
```
ema_fast, ema_slow, ema_diff, ema_diff_pct, ema_trend
+ lag features
```

### 2. Greeks Features (14)
Options Greeks for ATM calls and puts.
```
greeks_ce_atm_delta, greeks_ce_atm_gamma, greeks_ce_atm_theta,
greeks_ce_atm_vega, greeks_ce_atm_rho
greeks_pe_atm_delta, greeks_pe_atm_gamma, greeks_pe_atm_theta,
greeks_pe_atm_vega, greeks_pe_atm_rho
+ lag features
```

### 3. IV Features (10)
Implied volatility indicators.
```
avg_iv, iv_spread, iv_percentile, iv_rank
iv_mean_12, iv_change
+ lag features
```

### 4. PCR Features (8)
Put-Call Ratio indicators.
```
pcr_oi, pcr_volume
pcr_mean_12, pcr_change
+ lag features
```

### 5. Regime Features (5)
HMM regime detection outputs.
```
regime_filled, regime_prob_up, regime_prob_down, 
regime_prob_side, regime_confidence
```

### 6. Time Features (7)
Time-based patterns.
```
hour, minute, day_of_week
time_since_open, time_to_close
is_first_hour, is_last_hour
```

### 7. Lag Features (28)
Historical values of key indicators.
```
{feature}_lag1  (5 minutes ago)
{feature}_lag3  (15 minutes ago)
{feature}_lag6  (30 minutes ago)
{feature}_lag12 (1 hour ago)
```

Features with lags:
- spot_return_1
- ema_diff
- avg_iv
- pcr_oi
- greeks_ce_atm_delta
- gex_net
- spot_volatility_1h

### 8. Signal Strength Features (5)
Indicators of signal quality.
```
crossover_strength    - Magnitude of EMA difference at crossover
ema_diff_momentum     - Rate of change of EMA difference
trend_duration        - How long in current trend
regime_confidence     - Max regime probability
volume_ratio          - Current vs average volume
```

### 9. Rolling Features (7)
Rolling window statistics.
```
return_mean_12, return_std_12, return_skew_12
iv_mean_12, iv_change
pcr_mean_12, pcr_change
```

---

## Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load base features from nifty_features_5min.csv         │
│     └── EMA, Greeks, IV, PCR, Regime, Returns               │
│                                                              │
│  2. Create time features                                     │
│     └── hour, minute, day_of_week, session indicators       │
│                                                              │
│  3. Create lag features                                      │
│     └── 1, 3, 6, 12 period lags for key indicators         │
│                                                              │
│  4. Create signal strength features                          │
│     └── crossover_strength, trend_duration, etc.            │
│                                                              │
│  5. Create rolling features                                  │
│     └── 12-period rolling mean, std, skew                   │
│                                                              │
│  6. Create target variable                                   │
│     └── 1 if trade profitable, 0 otherwise                  │
│                                                              │
│  7. Handle missing values                                    │
│     └── Forward fill, backward fill, then fill with 0       │
│                                                              │
│  8. Scale features                                           │
│     └── StandardScaler (mean=0, std=1)                      │
│                                                              │
│  9. Split data (time-based)                                  │
│     └── 70% train, 30% test                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

### Missing Value Handling
1. Forward fill (use previous value)
2. Backward fill (for initial NaN)
3. Fill remaining with 0

### Feature Scaling
- **Method**: StandardScaler
- **Formula**: `z = (x - mean) / std`
- **Fit on**: Training data only
- **Transform**: Both train and test

### Train/Test Split
- **Method**: Time-based split (not random)
- **Reason**: Avoid lookahead bias
- **Split**: First 70% for training, last 30% for testing

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `ml_dataset.pkl` | `data/processed/` | Pickled ML dataset |

### Dataset Contents

```python
{
    'X_train': DataFrame,        # Scaled training features
    'X_test': DataFrame,         # Scaled testing features
    'y_train': Series,           # Training labels
    'y_test': Series,            # Testing labels
    'X_train_unscaled': DataFrame,  # Original training features
    'X_test_unscaled': DataFrame,   # Original testing features
    'feature_names': List[str],  # List of feature names
    'scaler': StandardScaler,    # Fitted scaler
    'ml_df': DataFrame,          # Full ML DataFrame
    'target_distribution': Dict  # Class distribution
}
```

---

## How to Use

### Loading the Dataset

```python
import pickle

with open('data/processed/ml_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
```

### Training a Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## Considerations

### Class Imbalance
- 69% unprofitable vs 31% profitable
- Consider: SMOTE, class weights, threshold tuning

### Feature Selection
- 86 features may lead to overfitting
- Consider: Feature importance, PCA, recursive elimination

### Evaluation Metrics
- Accuracy may be misleading due to imbalance
- Focus on: Precision, Recall, F1, AUC-ROC

### Lookahead Bias
- Time-based split prevents data leakage
- No future information in features

---

## Next Steps

1. **Task 5.2**: Train multiple classifiers
2. **Task 5.3**: Evaluate and compare models
3. **Task 5.4**: Feature importance analysis
4. **Task 5.5**: Integrate ML predictions with strategy

---

## Glossary

| Term | Definition |
|------|------------|
| **Binary Classification** | Predicting one of two classes |
| **Target Variable** | What we're trying to predict (profitable/not) |
| **Feature** | Input variable used for prediction |
| **Lag Feature** | Past value of a variable |
| **StandardScaler** | Normalizes features to mean=0, std=1 |
| **Lookahead Bias** | Using future information to predict past |
