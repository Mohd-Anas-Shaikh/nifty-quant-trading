# Task 5.2: Model Training

## Overview

This document describes the training of two machine learning models for trade profitability prediction:
- **Model A**: XGBoost (Gradient Boosting)
- **Model B**: LSTM (Deep Learning)

---

## Model A: XGBoost

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting algorithm that:
- Builds decision trees sequentially
- Each tree corrects errors of previous trees
- Includes regularization to prevent overfitting
- Handles missing values automatically

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_depth` | 5 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `n_estimators` | 100 | Number of trees |
| `subsample` | 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Column sampling ratio |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1 | L2 regularization |

### Time-Series Cross-Validation

Used 5-fold time-series CV to ensure:
- Training data always comes before validation data
- No lookahead bias
- Realistic performance estimation

```
Fold 1: Train [0:49]    → Val [49:90]
Fold 2: Train [0:90]    → Val [90:131]
Fold 3: Train [0:131]   → Val [131:172]
Fold 4: Train [0:172]   → Val [172:213]
Fold 5: Train [0:213]   → Val [213:248]
```

### CV Results

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.6146 | ±0.0730 |
| Precision | 0.1500 | ±0.2603 |
| Recall | 0.0468 | ±0.0596 |
| F1 Score | 0.0645 | ±0.0918 |
| AUC | 0.4385 | ±0.0739 |

### Test Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5701 |
| **Precision** | 0.3684 |
| **Recall** | 0.1707 |
| **F1 Score** | 0.2333 |
| **AUC** | 0.5916 |

### Top 5 Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `greeks_ce_atm_delta_lag12` | 0.0435 |
| 2 | `hour` | 0.0340 |
| 3 | `greeks_pe_atm_delta` | 0.0323 |
| 4 | `fut_dte` | 0.0286 |
| 5 | `delta_neutral_ratio` | 0.0252 |

---

## Model B: LSTM

### What is LSTM?

LSTM (Long Short-Term Memory) is a type of recurrent neural network that:
- Processes sequential data (time series)
- Remembers long-term dependencies
- Uses gates to control information flow
- Suitable for pattern recognition in sequences

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Layer                                                 │
│  └── Shape: (10, 86)                                        │
│      └── 10 time steps (candles)                            │
│      └── 86 features per candle                             │
│                                                              │
│  LSTM Layer                                                  │
│  └── 64 units                                               │
│  └── Parameters: 38,656                                     │
│                                                              │
│  Dropout Layer                                               │
│  └── Rate: 0.3 (30% dropout)                                │
│                                                              │
│  Dense Layer                                                 │
│  └── 32 units, ReLU activation                              │
│  └── Parameters: 2,080                                      │
│                                                              │
│  Dropout Layer                                               │
│  └── Rate: 0.2 (20% dropout)                                │
│                                                              │
│  Output Layer                                                │
│  └── 1 unit, Sigmoid activation                             │
│  └── Parameters: 33                                         │
│                                                              │
│  Total Parameters: 40,769                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Sequence Length | 10 candles |
| Batch Size | 16 |
| Max Epochs | 50 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Early Stopping | Patience=10 |

### Training Progress

- Model stopped early at epoch 11 (early stopping triggered)
- Training accuracy reached ~78%
- Validation accuracy peaked at ~61%
- Signs of overfitting (training >> validation)

### Test Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6082 |
| **Precision** | 0.0000 |
| **Recall** | 0.0000 |
| **F1 Score** | 0.0000 |
| **AUC** | 0.4732 |

**Note**: LSTM predicted all samples as class 0 (unprofitable), resulting in 0 precision/recall for class 1.

---

## Model Comparison

| Metric | XGBoost | LSTM | Winner |
|--------|---------|------|--------|
| Accuracy | 0.5701 | 0.6082 | LSTM |
| Precision | 0.3684 | 0.0000 | **XGBoost** |
| Recall | 0.1707 | 0.0000 | **XGBoost** |
| F1 Score | 0.2333 | 0.0000 | **XGBoost** |
| AUC | 0.5916 | 0.4732 | **XGBoost** |

### Analysis

1. **XGBoost outperforms LSTM** for this task
2. **LSTM overfits** - high training accuracy but poor generalization
3. **Small dataset** (355 samples) limits deep learning effectiveness
4. **Class imbalance** (69% unprofitable) affects both models

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `xgboost_model.pkl` | `data/processed/` | Trained XGBoost model |
| `lstm_model.keras` | `data/processed/` | Trained LSTM model |
| `lstm_model.pkl` | `data/processed/` | LSTM metadata |

---

## How to Use

### Loading XGBoost Model

```python
import pickle

with open('data/processed/xgboost_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)[:, 1]
```

### Loading LSTM Model

```python
from tensorflow import keras

model = keras.models.load_model('data/processed/lstm_model.keras')

# Create sequences (last 10 candles)
X_seq = X_new[-10:].reshape(1, 10, -1)
y_proba = model.predict(X_seq)
```

---

## Recommendations

1. **Use XGBoost** for production (better generalization)
2. **Address class imbalance** with SMOTE or class weights
3. **Collect more data** for LSTM to be effective
4. **Feature selection** to reduce overfitting
5. **Hyperparameter tuning** with grid search

---

## Glossary

| Term | Definition |
|------|------------|
| **XGBoost** | Extreme Gradient Boosting - ensemble tree method |
| **LSTM** | Long Short-Term Memory - recurrent neural network |
| **Time-Series CV** | Cross-validation respecting temporal order |
| **Overfitting** | Model memorizes training data, poor generalization |
| **AUC** | Area Under ROC Curve - classification performance |
| **Early Stopping** | Stop training when validation loss stops improving |
