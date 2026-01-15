#!/usr/bin/env python3
"""
Main script to run EMA Regime Strategy for Task 4.1

This script implements the 5/15 EMA crossover strategy with regime filter:
- LONG in Uptrend (regime +1)
- SHORT in Downtrend (regime -1)
- No trades in Sideways (regime 0)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.ema_regime_strategy import run_strategy

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, stats, trade_log = run_strategy(input_path, output_path)
    
    # Print sample trades
    print("\nSample Trades:")
    print(trade_log.head(10))
