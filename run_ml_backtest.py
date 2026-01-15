#!/usr/bin/env python3
"""
Main script to run ML-enhanced backtest for Task 5.3

Compares performance: Baseline vs XGBoost vs LSTM
Only takes trades when ML model predicts profitable (confidence > 0.5)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.ml_enhanced_backtest import run_ml_enhanced_backtest
import pickle

if __name__ == "__main__":
    data_path = Path("data/processed/nifty_features_5min.csv")
    ml_dataset_path = Path("data/processed/ml_dataset.pkl")
    xgb_model_path = Path("data/processed/xgboost_model.pkl")
    lstm_model_path = Path("data/processed/lstm_model")
    
    # Use lower threshold since model confidence is generally low
    results = run_ml_enhanced_backtest(
        data_path, ml_dataset_path, xgb_model_path, lstm_model_path,
        confidence_threshold=0.3
    )
    
    # Save results
    output_path = Path("data/processed/ml_backtest_comparison.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
