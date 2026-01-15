#!/usr/bin/env python3
"""
Main script to run outlier detection for Task 6.1

Identifies profitable trades beyond 3-sigma (Z-score > 3).
Analyzes features: regime, IV, ATR, time of day, Greeks, duration, EMA gap, PCR.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.outlier_detection import run_outlier_analysis
import pickle

if __name__ == "__main__":
    trade_log_path = Path("data/processed/trade_log.csv")
    features_path = Path("data/processed/nifty_features_5min.csv")
    
    results = run_outlier_analysis(trade_log_path, features_path, z_threshold=3.0)
    
    # Save results
    output_path = Path("data/processed/outlier_analysis.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
