#!/usr/bin/env python3
"""
Main script to calculate derived features for Task 2.3

This script creates derived features including:
- Average IV, IV Spread
- PCR (OI and Volume based)
- Futures Basis
- Returns
- Delta Neutral Ratio
- Gamma Exposure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.derived_features import add_derived_features_to_dataset

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = add_derived_features_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with derived features:")
    derived_cols = ['timestamp', 'spot_close', 'avg_iv', 'iv_spread', 'pcr_oi', 
                    'futures_basis_pct', 'spot_return_1', 'delta_neutral_ratio', 'gex_net']
    available_cols = [c for c in derived_cols if c in df.columns]
    print(df[available_cols].dropna().head(10))
