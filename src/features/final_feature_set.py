"""
Final Feature Set Module - Task 2.4

This module consolidates all features into the final dataset:
- EMA Indicators (Task 2.1)
- Options Greeks (Task 2.2)
- Derived Features (Task 2.3)
- Base data (spot, futures, options)

Generates the final nifty_features_5min.csv deliverable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class FinalFeatureSet:
    """
    Consolidates and validates the final feature set.
    """
    
    def __init__(self):
        self.feature_categories = {
            'base_spot': [],
            'base_futures': [],
            'base_options': [],
            'ema': [],
            'greeks': [],
            'iv': [],
            'pcr': [],
            'returns': [],
            'derived': [],
            'time': []
        }
    
    def categorize_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize all columns by feature type."""
        
        for col in df.columns:
            if col.startswith('spot_'):
                self.feature_categories['base_spot'].append(col)
            elif col.startswith('fut_'):
                self.feature_categories['base_futures'].append(col)
            elif col.startswith('opt_'):
                self.feature_categories['base_options'].append(col)
            elif col.startswith('ema_'):
                self.feature_categories['ema'].append(col)
            elif col.startswith('greeks_'):
                self.feature_categories['greeks'].append(col)
            elif 'iv' in col.lower() and not col.startswith('opt_'):
                self.feature_categories['iv'].append(col)
            elif 'pcr' in col.lower():
                self.feature_categories['pcr'].append(col)
            elif 'return' in col.lower() or 'volatility' in col.lower():
                self.feature_categories['returns'].append(col)
            elif col in ['timestamp', 'date', 'time', 'day_of_week', 'hour', 'minute']:
                self.feature_categories['time'].append(col)
            elif col.startswith('gex_') or col in ['delta_neutral_ratio', 'futures_basis_pct', 'price_momentum']:
                self.feature_categories['derived'].append(col)
        
        return self.feature_categories
    
    def validate_features(self, df: pd.DataFrame) -> Dict:
        """Validate that all required features are present."""
        
        required_features = {
            'EMA': ['ema_fast', 'ema_slow', 'ema_diff', 'ema_crossover', 'ema_trend'],
            'Greeks_CE': ['greeks_ce_atm_delta', 'greeks_ce_atm_gamma', 'greeks_ce_atm_theta', 
                         'greeks_ce_atm_vega', 'greeks_ce_atm_rho'],
            'Greeks_PE': ['greeks_pe_atm_delta', 'greeks_pe_atm_gamma', 'greeks_pe_atm_theta',
                         'greeks_pe_atm_vega', 'greeks_pe_atm_rho'],
            'IV': ['avg_iv', 'iv_spread'],
            'PCR': ['pcr_oi', 'pcr_volume'],
            'Returns': ['spot_return_1', 'spot_return_5', 'fut_return_1'],
            'Derived': ['futures_basis_pct', 'delta_neutral_ratio', 'gex_net']
        }
        
        validation = {'valid': True, 'missing': [], 'present': []}
        
        for category, features in required_features.items():
            for feat in features:
                if feat in df.columns:
                    validation['present'].append(feat)
                else:
                    validation['missing'].append(feat)
                    validation['valid'] = False
        
        return validation
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive feature summary."""
        
        categories = self.categorize_columns(df)
        
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'feature_counts': {
                'Base Spot': len(categories['base_spot']),
                'Base Futures': len(categories['base_futures']),
                'Base Options': len(categories['base_options']),
                'EMA Indicators': len(categories['ema']),
                'Greeks': len(categories['greeks']),
                'IV Features': len(categories['iv']),
                'PCR Features': len(categories['pcr']),
                'Returns & Volatility': len(categories['returns']),
                'Derived Features': len(categories['derived']),
                'Time Features': len(categories['time'])
            },
            'categories': categories
        }
        
        # Data quality
        summary['data_quality'] = {
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': round(df.isnull().sum().sum() / df.size * 100, 2),
            'rows_with_options': df['opt_ce_atm_ltp'].notna().sum() if 'opt_ce_atm_ltp' in df.columns else 0,
            'rows_with_greeks': df['greeks_ce_atm_delta'].notna().sum() if 'greeks_ce_atm_delta' in df.columns else 0
        }
        
        return summary
    
    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns in logical groups."""
        
        # Define column order
        time_cols = ['timestamp', 'date', 'time', 'day_of_week', 'hour', 'minute']
        spot_cols = sorted([c for c in df.columns if c.startswith('spot_')])
        fut_cols = sorted([c for c in df.columns if c.startswith('fut_')])
        ema_cols = sorted([c for c in df.columns if c.startswith('ema_')])
        opt_cols = sorted([c for c in df.columns if c.startswith('opt_')])
        greeks_cols = sorted([c for c in df.columns if c.startswith('greeks_')])
        
        # Derived features
        iv_cols = ['avg_iv', 'iv_spread', 'iv_percentile']
        pcr_cols = ['pcr_oi', 'pcr_volume']
        return_cols = sorted([c for c in df.columns if 'return' in c.lower() or 'volatility' in c.lower()])
        gex_cols = ['gex_ce', 'gex_pe', 'gex_net']
        other_derived = ['futures_basis_pct', 'delta_neutral_ratio', 'price_momentum']
        
        # Build ordered column list
        ordered_cols = []
        
        for col_group in [time_cols, spot_cols, fut_cols, ema_cols, opt_cols, greeks_cols,
                          iv_cols, pcr_cols, return_cols, gex_cols, other_derived]:
            for col in col_group:
                if col in df.columns and col not in ordered_cols:
                    ordered_cols.append(col)
        
        # Add any remaining columns
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        return df[ordered_cols]


def create_final_feature_set(input_path: Path, output_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Create the final feature set with all features consolidated.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save final output
        
    Returns:
        Tuple of (final DataFrame, summary dict)
    """
    print("=" * 70)
    print("FINAL FEATURE SET - Task 2.4")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize feature set handler
    feature_set = FinalFeatureSet()
    
    # Validate features
    print("\nValidating features...")
    validation = feature_set.validate_features(df)
    
    if validation['valid']:
        print("  ✓ All required features present")
    else:
        print(f"  ✗ Missing features: {validation['missing']}")
    
    # Get summary
    summary = feature_set.get_feature_summary(df)
    
    # Print feature counts
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    print(f"\nTotal Records: {summary['total_records']}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    print("\nFeature Categories:")
    for category, count in summary['feature_counts'].items():
        print(f"  {category}: {count} columns")
    
    print("\nData Quality:")
    print(f"  Missing Values: {summary['data_quality']['missing_values']} ({summary['data_quality']['missing_pct']}%)")
    print(f"  Rows with Options Data: {summary['data_quality']['rows_with_options']}")
    print(f"  Rows with Greeks: {summary['data_quality']['rows_with_greeks']}")
    
    # Reorder columns
    print("\nReordering columns...")
    df = feature_set.reorder_columns(df)
    
    # Save final dataset
    print(f"\nSaving final feature set to: {output_path}")
    df.to_csv(output_path, index=False)
    
    # Print column list by category
    print("\n" + "=" * 70)
    print("COLUMN LISTING BY CATEGORY")
    print("=" * 70)
    
    categories = summary['categories']
    
    print("\n[Time Features]")
    print(f"  {', '.join(categories['time'])}")
    
    print("\n[Spot Data]")
    print(f"  {', '.join(categories['base_spot'])}")
    
    print("\n[Futures Data]")
    print(f"  {', '.join(categories['base_futures'])}")
    
    print("\n[EMA Indicators]")
    print(f"  {', '.join(categories['ema'])}")
    
    print("\n[Options Data]")
    print(f"  {len(categories['base_options'])} columns (CE/PE for ATM±2 strikes)")
    
    print("\n[Greeks]")
    print(f"  {', '.join(categories['greeks'])}")
    
    print("\n[IV Features]")
    print(f"  {', '.join(categories['iv'])}")
    
    print("\n[PCR Features]")
    print(f"  {', '.join(categories['pcr'])}")
    
    print("\n[Returns & Volatility]")
    print(f"  {', '.join(categories['returns'])}")
    
    print("\n[Derived Features]")
    print(f"  {', '.join(categories['derived'])}")
    
    print("\n" + "=" * 70)
    print("FINAL FEATURE SET COMPLETE")
    print("=" * 70)
    print(f"\nDeliverable: {output_path}")
    print(f"Records: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    return df, summary


def main():
    """Main function to create final feature set."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = create_final_feature_set(input_path, output_path)
    
    # Print sample
    print("\nSample of final dataset:")
    key_cols = ['timestamp', 'spot_close', 'ema_fast', 'ema_slow', 'ema_trend',
                'greeks_ce_atm_delta', 'avg_iv', 'pcr_oi', 'spot_return_1', 'gex_net']
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head(10))
    
    return df, summary


if __name__ == "__main__":
    main()
