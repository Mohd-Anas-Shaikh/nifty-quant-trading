#!/usr/bin/env python3
"""
Main script to create final feature set for Task 2.4

This script consolidates all features (EMA, Greeks, IV, derived)
into the final nifty_features_5min.csv deliverable.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.final_feature_set import create_final_feature_set

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = create_final_feature_set(input_path, output_path)
    
    # Print sample
    print("\nSample of final dataset:")
    key_cols = ['timestamp', 'spot_close', 'ema_fast', 'ema_slow', 'ema_trend',
                'greeks_ce_atm_delta', 'avg_iv', 'pcr_oi', 'spot_return_1', 'gex_net']
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head(10))
