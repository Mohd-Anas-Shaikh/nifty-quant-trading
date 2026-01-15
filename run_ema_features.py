#!/usr/bin/env python3
"""
Main script to add EMA indicators for Task 2.1

This script calculates EMA(5) and EMA(15) indicators
for trading signal generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.ema_indicators import add_ema_to_dataset

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_merged_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = add_ema_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with EMA:")
    print(df[['timestamp', 'spot_close', 'ema_fast', 'ema_slow', 
              'ema_diff', 'ema_crossover', 'ema_trend']].head(20))
