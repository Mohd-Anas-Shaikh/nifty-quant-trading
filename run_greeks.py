#!/usr/bin/env python3
"""
Main script to calculate Options Greeks for Task 2.2

This script calculates Delta, Gamma, Theta, Vega, and Rho
for ATM Call and Put options using Black-Scholes model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.options_greeks import add_greeks_to_dataset

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = add_greeks_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with Greeks:")
    greek_cols = ['timestamp', 'spot_close', 'opt_ce_atm_iv', 'opt_dte',
                  'greeks_ce_atm_delta', 'greeks_ce_atm_gamma', 'greeks_ce_atm_theta',
                  'greeks_pe_atm_delta', 'greeks_pe_atm_gamma', 'greeks_pe_atm_theta']
    available_cols = [c for c in greek_cols if c in df.columns]
    print(df[available_cols].dropna().head(10))
