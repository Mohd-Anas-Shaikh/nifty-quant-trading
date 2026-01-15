#!/usr/bin/env python3
"""
Main script to prepare ML dataset for Task 5.1

This script builds the ML dataset for binary classification:
- Target: 1 if trade is profitable, 0 otherwise
- Features: Engineered features, regime, time, lags, signal strength
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.problem_definition import prepare_ml_dataset

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    trade_log_path = Path("data/processed/trade_log.csv")
    
    result = prepare_ml_dataset(input_path, trade_log_path)
    
    # Print sample
    print("\nSample features (first 5 rows):")
    print(result['X_train'].head())
    
    # Save ML dataset
    import pickle
    output_path = Path("data/processed/ml_dataset.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nML dataset saved to: {output_path}")
