# NIFTY 50 Quantitative Trading System

A comprehensive quantitative trading system for NIFTY 50 index, featuring EMA crossover strategy with HMM regime detection and machine learning enhancement.

## Project Overview

This project implements a complete quantitative trading pipeline:

1. **Data Acquisition**: 5-minute OHLCV data for NIFTY 50 spot, futures, and options
2. **Feature Engineering**: 86+ features including EMAs, Greeks, IV, PCR, and derived metrics
3. **Regime Detection**: Hidden Markov Model classifying market into Uptrend/Sideways/Downtrend
4. **Trading Strategy**: 5/15 EMA crossover with regime filter
5. **ML Enhancement**: XGBoost and LSTM models for trade profitability prediction
6. **Outlier Analysis**: Statistical analysis of high-performance trades

## Key Results

### Strategy Performance

| Metric | Training (70%) | Testing (30%) | Full Period |
|--------|----------------|---------------|-------------|
| Total Trades | 259 | 105 | 364 |
| Win Rate | 28.19% | 37.14% | 30.77% |
| Total Return | -0.52% | +0.60% | +0.08% |
| Profit Factor | 0.92 | 1.25 | 1.01 |
| Max Drawdown | 0.98% | 0.53% | 1.11% |

### ML Model Performance

| Model | Accuracy | AUC | Precision | Recall |
|-------|----------|-----|-----------|--------|
| XGBoost | 0.5701 | 0.5916 | 0.3684 | 0.1707 |
| LSTM | 0.6082 | 0.4732 | 0.0000 | 0.0000 |

### Outlier Analysis Insights

- **2.75%** of trades are outliers (Z-score > 3)
- Outliers contribute **3,409%** of total profits
- Dominant pattern: **Downtrend + SHORT** (70%)
- Key distinguishing feature: **Duration** (81x longer than normal)

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nifty-quant-trading.git
cd nifty-quant-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting
- `tensorflow` - Deep learning (LSTM)
- `hmmlearn` - Hidden Markov Models
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Statistical analysis

## How to Run

### Option 1: Run Individual Scripts

```bash
# 1. Data Acquisition
python run_data_fetch.py

# 2. Data Cleaning
python run_data_cleaning.py

# 3. Data Merging
python run_data_merge.py

# 4. Feature Engineering
python run_ema_features.py
python run_greeks.py
python run_derived_features.py
python run_final_features.py

# 5. Regime Detection
python run_hmm_regime.py
python run_regime_viz.py

# 6. Strategy Backtest
python run_strategy.py
python run_backtest.py

# 7. ML Models
python run_ml_prep.py
python run_model_training.py
python run_ml_backtest.py

# 8. Outlier Analysis
python run_outlier_analysis.py
python run_pattern_recognition.py
python run_insights_summary.py
```

### Option 2: Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Open notebooks in order:
1. `01_data_acquisition.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_regime_detection.ipynb`
5. `05_baseline_strategy.ipynb`
6. `06_ml_models.ipynb`
7. `07_outlier_analysis.ipynb`

## Project Structure

```
nifty-quant-trading/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_regime_detection.ipynb
│   ├── 05_baseline_strategy.ipynb
│   ├── 06_ml_models.ipynb
│   └── 07_outlier_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── config.py           # Configuration settings
│   │   ├── nse_data_fetcher.py # Data fetching utilities
│   │   ├── data_cleaner.py     # Data cleaning utilities
│   │   ├── data_merger.py      # Data merging utilities
│   │   └── broker_api_interface.py
│   ├── features/
│   │   ├── ema_indicators.py   # EMA calculations
│   │   ├── options_greeks.py   # Black-Scholes Greeks
│   │   ├── derived_features.py # Derived features
│   │   └── final_feature_set.py
│   ├── regime/
│   │   ├── hmm_regime.py       # HMM regime detection
│   │   └── regime_visualization.py
│   ├── strategy/
│   │   ├── ema_regime_strategy.py  # Trading strategy
│   │   └── backtester.py       # Backtesting engine
│   ├── ml/
│   │   ├── problem_definition.py   # ML dataset builder
│   │   ├── model_training.py   # XGBoost & LSTM models
│   │   └── ml_enhanced_backtest.py
│   └── analysis/
│       ├── outlier_detection.py    # Z-score analysis
│       ├── pattern_recognition.py  # Statistical tests
│       └── insights_summary.py
├── models/
│   ├── hmm_regime_model.pkl    # Trained HMM model
│   ├── xgboost_model.pkl       # Trained XGBoost model
│   ├── lstm_model.keras        # Trained LSTM model
│   └── ml_dataset.pkl          # ML dataset
├── results/
│   ├── nifty_features_5min.csv # Final feature dataset
│   ├── trade_log.csv           # Trade log
│   ├── equity_curve.csv        # Equity curve
│   └── backtest_report.txt     # Backtest report
├── plots/
│   ├── regime_price_overlay.png
│   ├── transition_matrix_heatmap.png
│   ├── pnl_vs_duration_scatter.png
│   ├── feature_boxplots.png
│   ├── correlation_heatmap.png
│   └── time_distribution.png
├── docs/
│   ├── TASK_1_1_DATA_ACQUISITION.md
│   ├── TASK_1_2_DATA_CLEANING.md
│   ├── TASK_1_3_DATA_MERGING.md
│   ├── TASK_2_1_EMA_INDICATORS.md
│   ├── TASK_2_2_OPTIONS_GREEKS.md
│   ├── TASK_2_3_DERIVED_FEATURES.md
│   ├── TASK_2_4_FINAL_FEATURE_SET.md
│   ├── TASK_3_1_HMM_REGIME.md
│   ├── TASK_3_2_REGIME_VISUALIZATION.md
│   ├── TASK_4_1_STRATEGY_IMPLEMENTATION.md
│   ├── TASK_4_2_BACKTESTING.md
│   ├── TASK_5_1_PROBLEM_DEFINITION.md
│   ├── TASK_5_2_MODEL_TRAINING.md
│   ├── TASK_5_3_ML_ENHANCED_BACKTEST.md
│   ├── TASK_6_1_OUTLIER_DETECTION.md
│   ├── TASK_6_2_PATTERN_RECOGNITION.md
│   └── TASK_6_3_INSIGHTS_SUMMARY.md
├── requirements.txt
└── README.md
```

## Features

### Technical Indicators
- EMA (5-period, 15-period)
- EMA crossover signals
- EMA trend direction

### Options Greeks (Black-Scholes)
- Delta, Gamma, Theta, Vega, Rho
- For ATM Call and Put options

### Derived Features
- Average IV, IV Spread
- Put-Call Ratio (OI and Volume based)
- Futures Basis
- Returns (1-period, 5-period)
- Delta Neutral Ratio
- Gamma Exposure (GEX)

### Regime Features
- HMM-based regime classification
- Regime probabilities
- Transition matrix

## Strategy Rules

### Entry Conditions
- **LONG**: EMA(5) crosses above EMA(15) AND Regime = Uptrend (+1)
- **SHORT**: EMA(5) crosses below EMA(15) AND Regime = Downtrend (-1)
- **No Trade**: Regime = Sideways (0)

### Exit Conditions
- **LONG Exit**: EMA(5) crosses below EMA(15)
- **SHORT Exit**: EMA(5) crosses above EMA(15)

### Execution
- Enter at next candle open after signal
- Exit at next candle open after exit signal

## Key Insights

### From Outlier Analysis

1. **Duration is Key**: Outlier trades are held 81x longer than normal trades
2. **Regime Matters**: 70% of outliers occur in Downtrend regime
3. **Direction**: SHORT trades dominate outliers (70%)
4. **Entry Conditions Don't Matter**: IV, Greeks, PCR show no significant difference
5. **Trade Management**: Success comes from holding winners, not better entries

### Recommendations

1. Focus on Downtrend + SHORT combinations
2. Extend holding periods on strong trends
3. Use ML filter with XGBoost (confidence > 0.3)
4. Prefer afternoon entries (peak hour: 13:00)

## Documentation

Detailed documentation for each task is available in the `docs/` directory.

## Acknowledgments

- NSE for market data structure reference
- hmmlearn for HMM implementation
- XGBoost and TensorFlow teams
