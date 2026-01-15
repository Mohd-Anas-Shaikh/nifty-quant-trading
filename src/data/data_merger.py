"""
Data Merging Module - Task 1.3

This module merges cleaned spot, futures, and options data on timestamp
to create a unified dataset for analysis and trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
from pathlib import Path

from .config import PROCESSED_DATA_DIR, STRIKE_INTERVAL


class DataMerger:
    """
    Merges NIFTY spot, futures, and options data into a single dataset.
    
    The merged dataset contains:
    - Spot OHLCV data
    - Futures OHLCV, OI, basis, and contract info
    - Options data pivoted by strike offset and type (CE/PE)
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or PROCESSED_DATA_DIR
    
    def load_cleaned_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cleaned data from CSV files."""
        print("Loading cleaned data...")
        
        spot_df = pd.read_csv(self.data_dir / "nifty_spot_5min_cleaned.csv")
        futures_df = pd.read_csv(self.data_dir / "nifty_futures_5min_cleaned.csv")
        options_df = pd.read_csv(self.data_dir / "nifty_options_5min_cleaned.csv")
        
        # Convert timestamps
        spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
        futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
        options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
        
        print(f"  Spot: {len(spot_df)} records")
        print(f"  Futures: {len(futures_df)} records")
        print(f"  Options: {len(options_df)} records")
        
        return spot_df, futures_df, options_df
    
    def prepare_spot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare spot data with prefixed column names."""
        df = df.copy()
        
        # Rename columns with spot_ prefix
        rename_map = {
            'open': 'spot_open',
            'high': 'spot_high',
            'low': 'spot_low',
            'close': 'spot_close',
            'volume': 'spot_volume'
        }
        df = df.rename(columns=rename_map)
        
        return df
    
    def prepare_futures_data(self, df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare futures data with basis calculation."""
        df = df.copy()
        
        # Rename columns with fut_ prefix
        rename_map = {
            'open': 'fut_open',
            'high': 'fut_high',
            'low': 'fut_low',
            'close': 'fut_close',
            'volume': 'fut_volume',
            'open_interest': 'fut_oi',
            'close_adjusted': 'fut_close_adjusted',
            'contract_month': 'fut_contract_month',
            'days_to_expiry': 'fut_dte'
        }
        df = df.rename(columns=rename_map)
        
        # Merge with spot to calculate basis
        merged = df.merge(
            spot_df[['timestamp', 'spot_close']], 
            on='timestamp', 
            how='left'
        )
        
        # Calculate basis (futures - spot)
        merged['fut_basis'] = merged['fut_close'] - merged['spot_close']
        merged['fut_basis_pct'] = (merged['fut_basis'] / merged['spot_close']) * 100
        
        # Drop spot_close (will come from spot merge)
        merged = merged.drop(columns=['spot_close'])
        
        return merged
    
    def prepare_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot options data to create columns for each strike/type combination.
        
        Creates columns like:
        - opt_ce_atm_ltp, opt_ce_atm_iv, opt_ce_atm_oi, opt_ce_atm_vol
        - opt_pe_atm_ltp, opt_pe_atm_iv, opt_pe_atm_oi, opt_pe_atm_vol
        - opt_ce_atm_p1_ltp, opt_ce_atm_p1_iv, ... (ATM+1)
        - opt_ce_atm_m1_ltp, opt_ce_atm_m1_iv, ... (ATM-1)
        """
        df = df.copy()
        
        # Create strike offset labels
        def get_strike_label(offset):
            if offset == 0:
                return 'atm'
            elif offset > 0:
                return f'atm_p{offset}'  # Plus (above ATM)
            else:
                return f'atm_m{abs(offset)}'  # Minus (below ATM)
        
        df['strike_label'] = df['strike_offset'].apply(get_strike_label)
        
        # Pivot the data
        pivoted_data = []
        
        for timestamp in df['timestamp'].unique():
            ts_data = df[df['timestamp'] == timestamp]
            row = {'timestamp': timestamp}
            
            # Get expiry info from first row
            if len(ts_data) > 0:
                row['opt_expiry'] = ts_data['expiry'].iloc[0]
                row['opt_dte'] = ts_data['days_to_expiry'].iloc[0]
            
            for _, opt_row in ts_data.iterrows():
                opt_type = opt_row['option_type'].lower()  # ce or pe
                strike_label = opt_row['strike_label']
                
                # Create column names
                prefix = f"opt_{opt_type}_{strike_label}"
                
                row[f"{prefix}_strike"] = opt_row['strike']
                row[f"{prefix}_ltp"] = opt_row['ltp']
                row[f"{prefix}_iv"] = opt_row['iv']
                row[f"{prefix}_oi"] = opt_row['open_interest']
                row[f"{prefix}_vol"] = opt_row['volume']
            
            pivoted_data.append(row)
        
        return pd.DataFrame(pivoted_data)
    
    def calculate_derived_options_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional options-derived metrics."""
        df = df.copy()
        
        # ATM Straddle price (CE + PE at ATM)
        if 'opt_ce_atm_ltp' in df.columns and 'opt_pe_atm_ltp' in df.columns:
            df['opt_atm_straddle'] = df['opt_ce_atm_ltp'] + df['opt_pe_atm_ltp']
        
        # ATM IV average
        if 'opt_ce_atm_iv' in df.columns and 'opt_pe_atm_iv' in df.columns:
            df['opt_atm_iv_avg'] = (df['opt_ce_atm_iv'] + df['opt_pe_atm_iv']) / 2
        
        # Put-Call Ratio (OI based) for ATM
        if 'opt_ce_atm_oi' in df.columns and 'opt_pe_atm_oi' in df.columns:
            df['opt_atm_pcr_oi'] = df['opt_pe_atm_oi'] / df['opt_ce_atm_oi'].replace(0, np.nan)
        
        # Put-Call Ratio (Volume based) for ATM
        if 'opt_ce_atm_vol' in df.columns and 'opt_pe_atm_vol' in df.columns:
            df['opt_atm_pcr_vol'] = df['opt_pe_atm_vol'] / df['opt_ce_atm_vol'].replace(0, np.nan)
        
        # Total OI across all strikes
        ce_oi_cols = [c for c in df.columns if c.startswith('opt_ce_') and c.endswith('_oi')]
        pe_oi_cols = [c for c in df.columns if c.startswith('opt_pe_') and c.endswith('_oi')]
        
        if ce_oi_cols:
            df['opt_total_ce_oi'] = df[ce_oi_cols].sum(axis=1)
        if pe_oi_cols:
            df['opt_total_pe_oi'] = df[pe_oi_cols].sum(axis=1)
        
        if 'opt_total_ce_oi' in df.columns and 'opt_total_pe_oi' in df.columns:
            df['opt_total_pcr_oi'] = df['opt_total_pe_oi'] / df['opt_total_ce_oi'].replace(0, np.nan)
        
        return df
    
    def merge_all(self) -> pd.DataFrame:
        """
        Merge all datasets on timestamp.
        
        Returns a single DataFrame with:
        - Spot data (spot_*)
        - Futures data (fut_*)
        - Options data (opt_*)
        - Derived metrics
        """
        print("\n" + "=" * 60)
        print("Data Merging - Task 1.3")
        print("=" * 60)
        
        # Load data
        spot_df, futures_df, options_df = self.load_cleaned_data()
        
        # Prepare each dataset
        print("\nPreparing datasets...")
        spot_prepared = self.prepare_spot_data(spot_df)
        futures_prepared = self.prepare_futures_data(futures_df, spot_prepared)
        options_prepared = self.prepare_options_data(options_df)
        
        print(f"  Spot columns: {len(spot_prepared.columns)}")
        print(f"  Futures columns: {len(futures_prepared.columns)}")
        print(f"  Options columns: {len(options_prepared.columns)}")
        
        # Merge spot and futures (should have same timestamps)
        print("\nMerging spot and futures...")
        merged = spot_prepared.merge(futures_prepared, on='timestamp', how='inner')
        print(f"  After spot+futures merge: {len(merged)} records")
        
        # Merge with options (options has fewer timestamps - hourly samples)
        print("\nMerging with options...")
        merged = merged.merge(options_prepared, on='timestamp', how='left')
        print(f"  After options merge: {len(merged)} records")
        
        # Calculate derived metrics
        print("\nCalculating derived metrics...")
        merged = self.calculate_derived_options_metrics(merged)
        
        # Sort by timestamp
        merged = merged.sort_values('timestamp').reset_index(drop=True)
        
        # Add date and time components for easier filtering
        merged['date'] = merged['timestamp'].dt.date
        merged['time'] = merged['timestamp'].dt.time
        merged['day_of_week'] = merged['timestamp'].dt.dayofweek
        merged['hour'] = merged['timestamp'].dt.hour
        merged['minute'] = merged['timestamp'].dt.minute
        
        print(f"\nFinal merged dataset:")
        print(f"  Records: {len(merged)}")
        print(f"  Columns: {len(merged.columns)}")
        print(f"  Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
        
        return merged
    
    def save_merged_data(self, df: pd.DataFrame) -> Path:
        """Save merged dataset to CSV."""
        output_path = self.data_dir / "nifty_merged_5min.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved merged data to: {output_path}")
        return output_path
    
    def print_column_summary(self, df: pd.DataFrame):
        """Print summary of columns in merged dataset."""
        print("\n" + "=" * 60)
        print("MERGED DATASET COLUMN SUMMARY")
        print("=" * 60)
        
        # Group columns by prefix
        spot_cols = [c for c in df.columns if c.startswith('spot_')]
        fut_cols = [c for c in df.columns if c.startswith('fut_')]
        opt_cols = [c for c in df.columns if c.startswith('opt_')]
        other_cols = [c for c in df.columns if not any(c.startswith(p) for p in ['spot_', 'fut_', 'opt_'])]
        
        print(f"\nSpot columns ({len(spot_cols)}):")
        print(f"  {', '.join(spot_cols)}")
        
        print(f"\nFutures columns ({len(fut_cols)}):")
        print(f"  {', '.join(fut_cols)}")
        
        print(f"\nOptions columns ({len(opt_cols)}):")
        # Group by type for readability
        opt_ce = [c for c in opt_cols if '_ce_' in c]
        opt_pe = [c for c in opt_cols if '_pe_' in c]
        opt_other = [c for c in opt_cols if '_ce_' not in c and '_pe_' not in c]
        print(f"  CE options: {len(opt_ce)} columns")
        print(f"  PE options: {len(opt_pe)} columns")
        print(f"  Derived: {', '.join(opt_other)}")
        
        print(f"\nOther columns ({len(other_cols)}):")
        print(f"  {', '.join(other_cols)}")


def main():
    """Main function to merge all data."""
    merger = DataMerger()
    
    # Merge all data
    merged_df = merger.merge_all()
    
    # Print column summary
    merger.print_column_summary(merged_df)
    
    # Save merged data
    merger.save_merged_data(merged_df)
    
    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 3 rows)")
    print("=" * 60)
    print(merged_df[['timestamp', 'spot_close', 'fut_close', 'fut_basis', 
                     'opt_ce_atm_ltp', 'opt_pe_atm_ltp', 'opt_atm_straddle']].head(3))
    
    print("\n" + "=" * 60)
    print("DATA MERGING COMPLETE")
    print("=" * 60)
    
    return merged_df


if __name__ == "__main__":
    main()
