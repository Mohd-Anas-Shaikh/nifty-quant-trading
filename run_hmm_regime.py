#!/usr/bin/env python3
"""
Main script to run HMM Regime Detection for Task 3.1

This script trains a Hidden Markov Model to classify market regimes:
- Regime +1: Uptrend
- Regime -1: Downtrend
- Regime 0: Sideways
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.regime.hmm_regime import train_hmm_regime_detector

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, results, stats = train_hmm_regime_detector(input_path, output_path)
    
    # Print sample
    print("\nSample data with regimes:")
    regime_cols = ['timestamp', 'spot_close', 'spot_return_1', 'avg_iv', 
                   'regime', 'regime_label', 'regime_prob_up', 'regime_prob_down']
    available = [c for c in regime_cols if c in df.columns]
    print(df[available].dropna().head(15))
