"""
Derived Features Module - Task 2.3

This module creates derived features from the base data:
- Average IV
- IV Spread
- PCR (OI-based and Volume-based)
- Futures Basis (percentage)
- Returns (spot and futures)
- Delta Neutral Ratio
- Gamma Exposure
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class DerivedFeatures:
    """
    Calculator for derived trading features.
    
    These features combine multiple data points to create
    meaningful signals for trading strategies.
    """
    
    def __init__(self):
        self.features_added = []
    
    def calculate_average_iv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average IV = (Call IV + Put IV) / 2
        
        Average IV gives a balanced view of market's volatility expectation,
        smoothing out any skew between calls and puts.
        """
        df = df.copy()
        
        if 'opt_ce_atm_iv' in df.columns and 'opt_pe_atm_iv' in df.columns:
            df['avg_iv'] = (df['opt_ce_atm_iv'] + df['opt_pe_atm_iv']) / 2
            self.features_added.append('avg_iv')
        
        return df
    
    def calculate_iv_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IV Spread = Call IV - Put IV
        
        IV Spread (also called IV skew) indicates market sentiment:
        - Positive spread: Calls are more expensive (bullish sentiment)
        - Negative spread: Puts are more expensive (bearish/hedging demand)
        """
        df = df.copy()
        
        if 'opt_ce_atm_iv' in df.columns and 'opt_pe_atm_iv' in df.columns:
            df['iv_spread'] = df['opt_ce_atm_iv'] - df['opt_pe_atm_iv']
            self.features_added.append('iv_spread')
        
        return df
    
    def calculate_pcr_oi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PCR (OI-based) = Total Put OI / Total Call OI
        
        Put-Call Ratio based on Open Interest:
        - PCR > 1: More puts outstanding (bearish positioning or hedging)
        - PCR < 1: More calls outstanding (bullish positioning)
        - PCR = 1: Balanced market
        
        Uses total OI across all strikes (ATM ± 2).
        """
        df = df.copy()
        
        # Sum OI across all strikes for calls
        ce_oi_cols = [c for c in df.columns if c.startswith('opt_ce_') and c.endswith('_oi')]
        pe_oi_cols = [c for c in df.columns if c.startswith('opt_pe_') and c.endswith('_oi')]
        
        if ce_oi_cols and pe_oi_cols:
            total_ce_oi = df[ce_oi_cols].sum(axis=1)
            total_pe_oi = df[pe_oi_cols].sum(axis=1)
            
            df['pcr_oi'] = total_pe_oi / total_ce_oi.replace(0, np.nan)
            self.features_added.append('pcr_oi')
        
        return df
    
    def calculate_pcr_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PCR (Volume-based) = Total Put Volume / Total Call Volume
        
        Put-Call Ratio based on trading volume:
        - More responsive to current market activity than OI
        - Shows real-time sentiment shifts
        
        Uses total volume across all strikes (ATM ± 2).
        """
        df = df.copy()
        
        # Sum volume across all strikes
        ce_vol_cols = [c for c in df.columns if c.startswith('opt_ce_') and c.endswith('_vol')]
        pe_vol_cols = [c for c in df.columns if c.startswith('opt_pe_') and c.endswith('_vol')]
        
        if ce_vol_cols and pe_vol_cols:
            total_ce_vol = df[ce_vol_cols].sum(axis=1)
            total_pe_vol = df[pe_vol_cols].sum(axis=1)
            
            df['pcr_volume'] = total_pe_vol / total_ce_vol.replace(0, np.nan)
            self.features_added.append('pcr_volume')
        
        return df
    
    def calculate_futures_basis_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Futures Basis (%) = (Futures Close - Spot Close) / Spot Close × 100
        
        Basis as percentage of spot price:
        - Positive basis: Futures trading at premium (contango, normal)
        - Negative basis: Futures trading at discount (backwardation, rare)
        - Basis narrows as expiry approaches
        """
        df = df.copy()
        
        if 'fut_close' in df.columns and 'spot_close' in df.columns:
            df['futures_basis_pct'] = ((df['fut_close'] - df['spot_close']) / df['spot_close']) * 100
            self.features_added.append('futures_basis_pct')
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Returns for spot and futures.
        
        Returns are calculated as percentage change:
        - Simple returns: (P_t - P_{t-1}) / P_{t-1} × 100
        - Log returns: ln(P_t / P_{t-1}) × 100
        
        We calculate both 1-period and 5-period returns.
        """
        df = df.copy()
        
        # Spot returns
        if 'spot_close' in df.columns:
            # 1-period (5-minute) returns
            df['spot_return_1'] = df['spot_close'].pct_change() * 100
            self.features_added.append('spot_return_1')
            
            # 5-period (25-minute) returns
            df['spot_return_5'] = df['spot_close'].pct_change(periods=5) * 100
            self.features_added.append('spot_return_5')
            
            # Log returns (1-period)
            df['spot_log_return'] = np.log(df['spot_close'] / df['spot_close'].shift(1)) * 100
            self.features_added.append('spot_log_return')
        
        # Futures returns
        if 'fut_close' in df.columns:
            # 1-period returns
            df['fut_return_1'] = df['fut_close'].pct_change() * 100
            self.features_added.append('fut_return_1')
            
            # 5-period returns
            df['fut_return_5'] = df['fut_close'].pct_change(periods=5) * 100
            self.features_added.append('fut_return_5')
            
            # Log returns
            df['fut_log_return'] = np.log(df['fut_close'] / df['fut_close'].shift(1)) * 100
            self.features_added.append('fut_log_return')
        
        return df
    
    def calculate_delta_neutral_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Delta Neutral Ratio = |Call Delta| / |Put Delta|
        
        This ratio helps understand the hedge ratio needed for delta-neutral positions:
        - Ratio = 1: Equal deltas, balanced hedge
        - Ratio > 1: Call delta larger, need more puts to hedge
        - Ratio < 1: Put delta larger, need more calls to hedge
        """
        df = df.copy()
        
        if 'greeks_ce_atm_delta' in df.columns and 'greeks_pe_atm_delta' in df.columns:
            abs_ce_delta = df['greeks_ce_atm_delta'].abs()
            abs_pe_delta = df['greeks_pe_atm_delta'].abs()
            
            df['delta_neutral_ratio'] = abs_ce_delta / abs_pe_delta.replace(0, np.nan)
            self.features_added.append('delta_neutral_ratio')
        
        return df
    
    def calculate_gamma_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gamma Exposure = Spot Close × Gamma × Open Interest
        
        Gamma Exposure (GEX) measures the market maker's gamma position:
        - High positive GEX: Market makers are long gamma, will buy dips/sell rallies
        - High negative GEX: Market makers are short gamma, will amplify moves
        
        GEX affects market dynamics and can predict volatility.
        
        We calculate for both calls and puts, then net GEX.
        """
        df = df.copy()
        
        # Call Gamma Exposure
        if all(col in df.columns for col in ['spot_close', 'greeks_ce_atm_gamma', 'opt_ce_atm_oi']):
            df['gex_ce'] = df['spot_close'] * df['greeks_ce_atm_gamma'] * df['opt_ce_atm_oi']
            self.features_added.append('gex_ce')
        
        # Put Gamma Exposure (negative because puts have negative delta effect)
        if all(col in df.columns for col in ['spot_close', 'greeks_pe_atm_gamma', 'opt_pe_atm_oi']):
            df['gex_pe'] = -df['spot_close'] * df['greeks_pe_atm_gamma'] * df['opt_pe_atm_oi']
            self.features_added.append('gex_pe')
        
        # Net Gamma Exposure
        if 'gex_ce' in df.columns and 'gex_pe' in df.columns:
            df['gex_net'] = df['gex_ce'] + df['gex_pe']
            self.features_added.append('gex_net')
        
        return df
    
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional useful derived features.
        """
        df = df.copy()
        
        # Volatility of returns (rolling 12-period = 1 hour)
        if 'spot_return_1' in df.columns:
            df['spot_volatility_1h'] = df['spot_return_1'].rolling(window=12).std()
            self.features_added.append('spot_volatility_1h')
        
        # Price momentum (current price vs EMA)
        if 'spot_close' in df.columns and 'ema_slow' in df.columns:
            df['price_momentum'] = ((df['spot_close'] - df['ema_slow']) / df['ema_slow']) * 100
            self.features_added.append('price_momentum')
        
        # IV percentile (where current IV stands relative to recent history)
        if 'avg_iv' in df.columns:
            df['iv_percentile'] = df['avg_iv'].rolling(window=75).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 
                if x.max() != x.min() else 50
            )
            self.features_added.append('iv_percentile')
        
        return df
    
    def add_all_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all derived features to the DataFrame.
        """
        df = self.calculate_average_iv(df)
        df = self.calculate_iv_spread(df)
        df = self.calculate_pcr_oi(df)
        df = self.calculate_pcr_volume(df)
        df = self.calculate_futures_basis_pct(df)
        df = self.calculate_returns(df)
        df = self.calculate_delta_neutral_ratio(df)
        df = self.calculate_gamma_exposure(df)
        df = self.calculate_additional_features(df)
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for derived features."""
        summary = {
            'total_features_added': len(self.features_added),
            'features': self.features_added,
        }
        
        # Add statistics for key features
        key_features = ['avg_iv', 'iv_spread', 'pcr_oi', 'pcr_volume', 
                        'futures_basis_pct', 'spot_return_1', 'delta_neutral_ratio', 'gex_net']
        
        for feat in key_features:
            if feat in df.columns:
                summary[f'{feat}_mean'] = round(df[feat].mean(), 4)
                summary[f'{feat}_std'] = round(df[feat].std(), 4)
                summary[f'{feat}_min'] = round(df[feat].min(), 4)
                summary[f'{feat}_max'] = round(df[feat].max(), 4)
        
        return summary


def add_derived_features_to_dataset(input_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Add derived features to a dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output (optional)
        
    Returns:
        Tuple of (DataFrame with features, summary dict)
    """
    print("=" * 60)
    print("Derived Features - Task 2.3")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Records: {len(df)}")
    print(f"  Existing columns: {len(df.columns)}")
    
    # Initialize feature calculator
    feature_calc = DerivedFeatures()
    
    # Add all derived features
    print("\nCalculating derived features...")
    df = feature_calc.add_all_derived_features(df)
    
    # Get summary
    summary = feature_calc.get_feature_summary(df)
    
    print(f"\nFeatures Added: {summary['total_features_added']}")
    print("\nFeature Statistics:")
    
    print("\n  IV Features:")
    print(f"    Average IV - Mean: {summary.get('avg_iv_mean', 'N/A')}%, Std: {summary.get('avg_iv_std', 'N/A')}%")
    print(f"    IV Spread - Mean: {summary.get('iv_spread_mean', 'N/A')}%, Range: [{summary.get('iv_spread_min', 'N/A')}, {summary.get('iv_spread_max', 'N/A')}]")
    
    print("\n  PCR Features:")
    print(f"    PCR (OI) - Mean: {summary.get('pcr_oi_mean', 'N/A')}, Std: {summary.get('pcr_oi_std', 'N/A')}")
    print(f"    PCR (Volume) - Mean: {summary.get('pcr_volume_mean', 'N/A')}, Std: {summary.get('pcr_volume_std', 'N/A')}")
    
    print("\n  Basis & Returns:")
    print(f"    Futures Basis - Mean: {summary.get('futures_basis_pct_mean', 'N/A')}%")
    print(f"    Spot Return (1-period) - Mean: {summary.get('spot_return_1_mean', 'N/A')}%, Std: {summary.get('spot_return_1_std', 'N/A')}%")
    
    print("\n  Greeks-Derived:")
    print(f"    Delta Neutral Ratio - Mean: {summary.get('delta_neutral_ratio_mean', 'N/A')}")
    print(f"    Net Gamma Exposure - Mean: {summary.get('gex_net_mean', 'N/A')}")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        print(f"Total columns: {len(df.columns)}")
    
    print("\n" + "=" * 60)
    print("DERIVED FEATURES COMPLETE")
    print("=" * 60)
    
    return df, summary


def main():
    """Main function to add derived features."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, summary = add_derived_features_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with derived features:")
    derived_cols = ['timestamp', 'spot_close', 'avg_iv', 'iv_spread', 'pcr_oi', 
                    'futures_basis_pct', 'spot_return_1', 'delta_neutral_ratio', 'gex_net']
    available_cols = [c for c in derived_cols if c in df.columns]
    print(df[available_cols].dropna().head(10))
    
    return df, summary


if __name__ == "__main__":
    main()
