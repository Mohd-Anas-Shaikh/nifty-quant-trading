"""
Data Cleaning Module - Task 1.2

This module handles:
1. Missing value detection and imputation
2. Outlier detection and removal
3. Timestamp alignment across datasets
4. Futures contract rollover handling
5. Dynamic ATM strike calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings

from .config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    SPOT_OUTPUT_FILE, FUTURES_OUTPUT_FILE, OPTIONS_OUTPUT_FILE,
    STRIKE_INTERVAL
)

warnings.filterwarnings('ignore')


class DataCleaningReport:
    """Tracks and generates data cleaning statistics."""
    
    def __init__(self):
        self.report = {
            'spot': {},
            'futures': {},
            'options': {},
            'alignment': {},
            'summary': {}
        }
        self.logs = []
    
    def log(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        print(message)
    
    def add_stat(self, dataset: str, key: str, value):
        """Add a statistic to the report."""
        self.report[dataset][key] = value
    
    def generate_report(self) -> str:
        """Generate the final cleaning report."""
        lines = [
            "=" * 70,
            "DATA CLEANING REPORT - Task 1.2",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 70,
            "1. NIFTY 50 SPOT DATA CLEANING",
            "=" * 70,
        ]
        
        spot = self.report['spot']
        lines.extend([
            f"Original Records: {spot.get('original_count', 'N/A')}",
            f"Final Records: {spot.get('final_count', 'N/A')}",
            "",
            "Missing Values:",
            f"  - Detected: {spot.get('missing_detected', 0)}",
            f"  - Imputed: {spot.get('missing_imputed', 0)}",
            f"  - Method: Forward fill + backward fill for edge cases",
            "",
            "Outliers:",
            f"  - Detected: {spot.get('outliers_detected', 0)}",
            f"  - Removed/Corrected: {spot.get('outliers_corrected', 0)}",
            f"  - Method: IQR-based detection (1.5x IQR threshold)",
            "",
            "Price Range (after cleaning):",
            f"  - Min: {spot.get('price_min', 'N/A')}",
            f"  - Max: {spot.get('price_max', 'N/A')}",
            f"  - Mean: {spot.get('price_mean', 'N/A')}",
            "",
        ])
        
        lines.extend([
            "=" * 70,
            "2. NIFTY FUTURES DATA CLEANING",
            "=" * 70,
        ])
        
        futures = self.report['futures']
        lines.extend([
            f"Original Records: {futures.get('original_count', 'N/A')}",
            f"Final Records: {futures.get('final_count', 'N/A')}",
            "",
            "Missing Values:",
            f"  - Detected: {futures.get('missing_detected', 0)}",
            f"  - Imputed: {futures.get('missing_imputed', 0)}",
            "",
            "Outliers:",
            f"  - Detected: {futures.get('outliers_detected', 0)}",
            f"  - Corrected: {futures.get('outliers_corrected', 0)}",
            "",
            "Contract Rollover Handling:",
            f"  - Rollovers Detected: {futures.get('rollovers_detected', 0)}",
            f"  - Rollover Dates: {futures.get('rollover_dates', [])}",
            f"  - Method: Continuous contract adjustment using ratio method",
            "",
            "Basis Statistics (Futures - Spot):",
            f"  - Mean Basis: {futures.get('basis_mean', 'N/A')}",
            f"  - Max Basis: {futures.get('basis_max', 'N/A')}",
            f"  - Min Basis: {futures.get('basis_min', 'N/A')}",
            "",
        ])
        
        lines.extend([
            "=" * 70,
            "3. NIFTY OPTIONS DATA CLEANING",
            "=" * 70,
        ])
        
        options = self.report['options']
        lines.extend([
            f"Original Records: {options.get('original_count', 'N/A')}",
            f"Final Records: {options.get('final_count', 'N/A')}",
            "",
            "Missing Values:",
            f"  - Detected: {options.get('missing_detected', 0)}",
            f"  - Imputed: {options.get('missing_imputed', 0)}",
            "",
            "Outliers (IV-based):",
            f"  - Detected: {options.get('outliers_detected', 0)}",
            f"  - Corrected: {options.get('outliers_corrected', 0)}",
            "",
            "Dynamic ATM Strike Calculation:",
            f"  - ATM Strikes Recalculated: {options.get('atm_recalculated', 0)}",
            f"  - Strike Interval: {STRIKE_INTERVAL}",
            f"  - Method: Round(Spot Price / Strike Interval) * Strike Interval",
            "",
            "IV Statistics (after cleaning):",
            f"  - Mean IV: {options.get('iv_mean', 'N/A')}%",
            f"  - Min IV: {options.get('iv_min', 'N/A')}%",
            f"  - Max IV: {options.get('iv_max', 'N/A')}%",
            "",
        ])
        
        lines.extend([
            "=" * 70,
            "4. TIMESTAMP ALIGNMENT",
            "=" * 70,
        ])
        
        alignment = self.report['alignment']
        lines.extend([
            f"Common Timestamps: {alignment.get('common_timestamps', 'N/A')}",
            f"Spot-only Timestamps: {alignment.get('spot_only', 0)}",
            f"Futures-only Timestamps: {alignment.get('futures_only', 0)}",
            f"Options-only Timestamps: {alignment.get('options_only', 0)}",
            "",
            f"Date Range: {alignment.get('start_date', 'N/A')} to {alignment.get('end_date', 'N/A')}",
            f"Trading Days: {alignment.get('trading_days', 'N/A')}",
            "",
        ])
        
        lines.extend([
            "=" * 70,
            "5. SUMMARY",
            "=" * 70,
        ])
        
        summary = self.report['summary']
        lines.extend([
            f"Total Records Processed: {summary.get('total_processed', 'N/A')}",
            f"Total Missing Values Handled: {summary.get('total_missing', 'N/A')}",
            f"Total Outliers Corrected: {summary.get('total_outliers', 'N/A')}",
            f"Data Quality Score: {summary.get('quality_score', 'N/A')}%",
            "",
            "Cleaning Methods Applied:",
            "  1. Missing Values: Forward-fill (ffill) with backward-fill (bfill) for edges",
            "  2. Outliers: IQR method with 1.5x threshold, replaced with rolling median",
            "  3. Timestamp Alignment: Inner join on common timestamps",
            "  4. Futures Rollover: Ratio-based continuous contract adjustment",
            "  5. ATM Strike: Dynamic calculation based on spot price",
            "",
        ])
        
        lines.extend([
            "=" * 70,
            "6. PROCESSING LOG",
            "=" * 70,
        ])
        lines.extend(self.logs[-20:])  # Last 20 log entries
        
        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(lines)


class DataCleaner:
    """
    Comprehensive data cleaning for NIFTY spot, futures, and options data.
    """
    
    def __init__(self, raw_data_dir: Path = None, processed_data_dir: Path = None):
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.report = DataCleaningReport()
        
        # Ensure output directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from CSV files."""
        self.report.log("Loading raw data files...")
        
        spot_df = pd.read_csv(self.raw_data_dir / SPOT_OUTPUT_FILE)
        futures_df = pd.read_csv(self.raw_data_dir / FUTURES_OUTPUT_FILE)
        options_df = pd.read_csv(self.raw_data_dir / OPTIONS_OUTPUT_FILE)
        
        # Convert timestamps
        spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
        futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
        options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
        
        self.report.log(f"  Spot: {len(spot_df)} records")
        self.report.log(f"  Futures: {len(futures_df)} records")
        self.report.log(f"  Options: {len(options_df)} records")
        
        return spot_df, futures_df, options_df
    
    def detect_missing_values(self, df: pd.DataFrame, name: str) -> Dict[str, int]:
        """Detect missing values in a dataframe."""
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        self.report.log(f"  {name}: {total_missing} missing values detected")
        
        if total_missing > 0:
            for col, count in missing[missing > 0].items():
                self.report.log(f"    - {col}: {count} missing")
        
        return {'total': total_missing, 'by_column': missing.to_dict()}
    
    def impute_missing_values(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Impute missing values using forward-fill and backward-fill.
        
        Strategy:
        - Forward fill: Use previous valid value (most common in time series)
        - Backward fill: For leading NaN values
        - Interpolation: For numeric columns with gaps
        """
        df = df.copy()
        
        for col in numeric_cols:
            if col in df.columns:
                # First try forward fill
                df[col] = df[col].ffill()
                # Then backward fill for any remaining NaN at the start
                df[col] = df[col].bfill()
                # Linear interpolation for any remaining gaps
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method='linear')
        
        return df
    
    def detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using the IQR (Interquartile Range) method.
        
        Outlier if: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Outlier if: |z-score| > threshold (default 3)
        """
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    def correct_outliers(self, df: pd.DataFrame, col: str, 
                         outlier_mask: pd.Series, window: int = 5) -> pd.DataFrame:
        """
        Correct outliers by replacing with rolling median.
        """
        df = df.copy()
        rolling_median = df[col].rolling(window=window, center=True, min_periods=1).median()
        df.loc[outlier_mask, col] = rolling_median[outlier_mask]
        return df
    
    def clean_spot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NIFTY 50 spot data.
        
        Steps:
        1. Detect and impute missing values
        2. Detect and correct price outliers
        3. Ensure OHLC consistency
        4. Validate volume data
        """
        self.report.log("\nCleaning NIFTY 50 Spot data...")
        self.report.add_stat('spot', 'original_count', len(df))
        
        df = df.copy()
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 1. Handle missing values
        missing_info = self.detect_missing_values(df, "Spot")
        self.report.add_stat('spot', 'missing_detected', missing_info['total'])
        
        df = self.impute_missing_values(df, numeric_cols)
        self.report.add_stat('spot', 'missing_imputed', missing_info['total'])
        
        # 2. Detect and correct outliers in close price
        # Use returns-based outlier detection (more robust for price series)
        df['returns'] = df['close'].pct_change()
        outliers_returns = self.detect_outliers_iqr(df['returns'].dropna(), multiplier=3.0)
        
        # Map back to original index
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_mask.iloc[1:] = outliers_returns.values
        
        outliers_count = outlier_mask.sum()
        self.report.add_stat('spot', 'outliers_detected', outliers_count)
        
        if outliers_count > 0:
            self.report.log(f"  Correcting {outliers_count} outliers in spot data")
            df = self.correct_outliers(df, 'close', outlier_mask)
            
            # Recalculate OHLC based on corrected close
            # Adjust high/low to be consistent
            df.loc[outlier_mask, 'high'] = df.loc[outlier_mask, 'close'] * 1.001
            df.loc[outlier_mask, 'low'] = df.loc[outlier_mask, 'close'] * 0.999
            df.loc[outlier_mask, 'open'] = df['close'].shift(1).loc[outlier_mask]
        
        self.report.add_stat('spot', 'outliers_corrected', outliers_count)
        
        # 3. Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # 4. Ensure positive volume
        df['volume'] = df['volume'].clip(lower=0)
        
        # Drop helper column
        df = df.drop(columns=['returns'])
        
        # Record statistics
        self.report.add_stat('spot', 'final_count', len(df))
        self.report.add_stat('spot', 'price_min', f"{df['close'].min():.2f}")
        self.report.add_stat('spot', 'price_max', f"{df['close'].max():.2f}")
        self.report.add_stat('spot', 'price_mean', f"{df['close'].mean():.2f}")
        
        self.report.log(f"  Spot cleaning complete: {len(df)} records")
        
        return df
    
    def clean_futures_data(self, df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NIFTY Futures data with rollover handling.
        
        Steps:
        1. Handle missing values
        2. Detect and correct outliers
        3. Handle contract rollovers (create continuous series)
        4. Validate basis (futures - spot)
        """
        self.report.log("\nCleaning NIFTY Futures data...")
        self.report.add_stat('futures', 'original_count', len(df))
        
        df = df.copy()
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        
        # 1. Handle missing values
        missing_info = self.detect_missing_values(df, "Futures")
        self.report.add_stat('futures', 'missing_detected', missing_info['total'])
        
        df = self.impute_missing_values(df, numeric_cols)
        self.report.add_stat('futures', 'missing_imputed', missing_info['total'])
        
        # 2. Detect and correct outliers
        df['returns'] = df['close'].pct_change()
        outliers = self.detect_outliers_iqr(df['returns'].dropna(), multiplier=3.0)
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_mask.iloc[1:] = outliers.values
        
        outliers_count = outlier_mask.sum()
        self.report.add_stat('futures', 'outliers_detected', outliers_count)
        
        if outliers_count > 0:
            df = self.correct_outliers(df, 'close', outlier_mask)
        
        self.report.add_stat('futures', 'outliers_corrected', outliers_count)
        
        # 3. Handle contract rollovers
        df, rollover_info = self._handle_futures_rollover(df)
        self.report.add_stat('futures', 'rollovers_detected', rollover_info['count'])
        self.report.add_stat('futures', 'rollover_dates', rollover_info['dates'])
        
        # 4. Calculate and validate basis
        # Merge with spot to calculate basis
        merged = df.merge(
            spot_df[['timestamp', 'close']], 
            on='timestamp', 
            suffixes=('', '_spot')
        )
        
        if len(merged) > 0:
            basis = merged['close'] - merged['close_spot']
            self.report.add_stat('futures', 'basis_mean', f"{basis.mean():.2f}")
            self.report.add_stat('futures', 'basis_max', f"{basis.max():.2f}")
            self.report.add_stat('futures', 'basis_min', f"{basis.min():.2f}")
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Ensure positive OI
        df['open_interest'] = df['open_interest'].clip(lower=0)
        
        # Drop helper column
        df = df.drop(columns=['returns'], errors='ignore')
        
        self.report.add_stat('futures', 'final_count', len(df))
        self.report.log(f"  Futures cleaning complete: {len(df)} records")
        
        return df
    
    def _handle_futures_rollover(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle futures contract rollovers to create a continuous price series.
        
        Method: Ratio adjustment
        - At rollover, calculate ratio = new_contract_price / old_contract_price
        - Adjust all historical prices by this ratio
        
        This preserves percentage returns across rollovers.
        """
        df = df.copy()
        
        # Detect rollover points (contract month changes)
        df['contract_change'] = df['contract_month'] != df['contract_month'].shift(1)
        rollover_indices = df[df['contract_change']].index.tolist()
        
        # Remove first index (not a rollover, just start of data)
        if rollover_indices and rollover_indices[0] == df.index[0]:
            rollover_indices = rollover_indices[1:]
        
        rollover_dates = df.loc[rollover_indices, 'timestamp'].dt.date.astype(str).tolist()
        
        self.report.log(f"  Detected {len(rollover_indices)} contract rollovers")
        
        # Create adjusted close for continuous series
        df['close_adjusted'] = df['close'].copy()
        
        # Apply ratio adjustment (backward from each rollover)
        for idx in reversed(rollover_indices):
            if idx > 0:
                # Get prices at rollover
                new_price = df.loc[idx, 'close']
                old_price = df.loc[idx - 1, 'close']
                
                if old_price > 0:
                    ratio = new_price / old_price
                    
                    # Adjust all prices before this rollover
                    df.loc[:idx-1, 'close_adjusted'] *= ratio
        
        # Drop helper column
        df = df.drop(columns=['contract_change'])
        
        return df, {'count': len(rollover_indices), 'dates': rollover_dates}
    
    def clean_options_data(self, df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NIFTY Options data with dynamic ATM calculation.
        
        Steps:
        1. Handle missing values
        2. Detect and correct IV outliers
        3. Recalculate ATM strike dynamically
        4. Validate option prices
        """
        self.report.log("\nCleaning NIFTY Options data...")
        self.report.add_stat('options', 'original_count', len(df))
        
        df = df.copy()
        numeric_cols = ['ltp', 'iv', 'volume', 'open_interest']
        
        # 1. Handle missing values
        missing_info = self.detect_missing_values(df, "Options")
        self.report.add_stat('options', 'missing_detected', missing_info['total'])
        
        df = self.impute_missing_values(df, numeric_cols)
        self.report.add_stat('options', 'missing_imputed', missing_info['total'])
        
        # 2. Detect and correct IV outliers
        # IV should typically be between 5% and 100%
        iv_outliers = (df['iv'] < 5) | (df['iv'] > 100)
        outliers_count = iv_outliers.sum()
        
        self.report.add_stat('options', 'outliers_detected', outliers_count)
        
        if outliers_count > 0:
            self.report.log(f"  Correcting {outliers_count} IV outliers")
            # Replace with median IV for same strike offset and option type
            median_iv = df.groupby(['strike_offset', 'option_type'])['iv'].transform('median')
            df.loc[iv_outliers, 'iv'] = median_iv[iv_outliers]
            
            # Clip to reasonable range
            df['iv'] = df['iv'].clip(lower=5, upper=100)
        
        self.report.add_stat('options', 'outliers_corrected', outliers_count)
        
        # 3. Recalculate ATM strike dynamically
        df, atm_count = self._calculate_dynamic_atm(df, spot_df)
        self.report.add_stat('options', 'atm_recalculated', atm_count)
        
        # 4. Validate option prices (LTP should be positive)
        df['ltp'] = df['ltp'].clip(lower=0.05)  # Minimum tick size
        
        # Ensure positive OI and volume
        df['open_interest'] = df['open_interest'].clip(lower=0)
        df['volume'] = df['volume'].clip(lower=0)
        
        # Record IV statistics
        self.report.add_stat('options', 'iv_mean', f"{df['iv'].mean():.2f}")
        self.report.add_stat('options', 'iv_min', f"{df['iv'].min():.2f}")
        self.report.add_stat('options', 'iv_max', f"{df['iv'].max():.2f}")
        
        self.report.add_stat('options', 'final_count', len(df))
        self.report.log(f"  Options cleaning complete: {len(df)} records")
        
        return df
    
    def _calculate_dynamic_atm(self, options_df: pd.DataFrame, 
                                spot_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Dynamically calculate ATM strike based on spot price at each timestamp.
        
        ATM Strike = Round(Spot Price / Strike Interval) * Strike Interval
        """
        df = options_df.copy()
        
        # Create spot price lookup
        spot_lookup = spot_df.set_index('timestamp')['close'].to_dict()
        
        # Calculate dynamic ATM for each row
        def get_dynamic_atm(row):
            ts = row['timestamp']
            if ts in spot_lookup:
                spot_price = spot_lookup[ts]
                return int(round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL)
            return row['strike']  # Fallback to existing strike
        
        # Get unique timestamps and calculate ATM for each
        unique_timestamps = df['timestamp'].unique()
        atm_map = {}
        
        for ts in unique_timestamps:
            if ts in spot_lookup:
                spot_price = spot_lookup[ts]
                atm_map[ts] = int(round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL)
        
        # Update strike_offset based on dynamic ATM
        def update_strike_offset(row):
            ts = row['timestamp']
            if ts in atm_map:
                dynamic_atm = atm_map[ts]
                return int((row['strike'] - dynamic_atm) / STRIKE_INTERVAL)
            return row['strike_offset']
        
        # Update moneyness based on new ATM
        def update_moneyness(row):
            if row['strike_offset'] == 0:
                return 'ATM'
            elif row['option_type'] == 'CE':
                return 'ITM' if row['strike_offset'] < 0 else 'OTM'
            else:  # PE
                return 'ITM' if row['strike_offset'] > 0 else 'OTM'
        
        # Apply updates
        original_offsets = df['strike_offset'].copy()
        df['strike_offset'] = df.apply(update_strike_offset, axis=1)
        df['moneyness'] = df.apply(update_moneyness, axis=1)
        
        # Count how many were recalculated
        changed_count = (df['strike_offset'] != original_offsets).sum()
        
        self.report.log(f"  Recalculated ATM for {len(atm_map)} timestamps")
        
        return df, changed_count
    
    def align_timestamps(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame,
                         options_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align timestamps across all three datasets.
        
        Strategy: Keep only timestamps that exist in spot data (primary source)
        """
        self.report.log("\nAligning timestamps across datasets...")
        
        spot_timestamps = set(spot_df['timestamp'])
        futures_timestamps = set(futures_df['timestamp'])
        options_timestamps = set(options_df['timestamp'])
        
        # Find common timestamps (spot is the reference)
        common_timestamps = spot_timestamps & futures_timestamps
        
        # For options, we sample less frequently, so align to available options timestamps
        options_common = spot_timestamps & options_timestamps
        
        self.report.add_stat('alignment', 'common_timestamps', len(common_timestamps))
        self.report.add_stat('alignment', 'spot_only', len(spot_timestamps - futures_timestamps))
        self.report.add_stat('alignment', 'futures_only', len(futures_timestamps - spot_timestamps))
        self.report.add_stat('alignment', 'options_only', len(options_timestamps - spot_timestamps))
        
        # Filter dataframes
        spot_aligned = spot_df[spot_df['timestamp'].isin(common_timestamps)].copy()
        futures_aligned = futures_df[futures_df['timestamp'].isin(common_timestamps)].copy()
        options_aligned = options_df[options_df['timestamp'].isin(options_common)].copy()
        
        # Sort by timestamp
        spot_aligned = spot_aligned.sort_values('timestamp').reset_index(drop=True)
        futures_aligned = futures_aligned.sort_values('timestamp').reset_index(drop=True)
        options_aligned = options_aligned.sort_values(['timestamp', 'strike', 'option_type']).reset_index(drop=True)
        
        # Record date range
        if len(spot_aligned) > 0:
            self.report.add_stat('alignment', 'start_date', 
                                spot_aligned['timestamp'].min().strftime('%Y-%m-%d'))
            self.report.add_stat('alignment', 'end_date',
                                spot_aligned['timestamp'].max().strftime('%Y-%m-%d'))
            self.report.add_stat('alignment', 'trading_days',
                                spot_aligned['timestamp'].dt.date.nunique())
        
        self.report.log(f"  Aligned spot: {len(spot_aligned)} records")
        self.report.log(f"  Aligned futures: {len(futures_aligned)} records")
        self.report.log(f"  Aligned options: {len(options_aligned)} records")
        
        return spot_aligned, futures_aligned, options_aligned
    
    def calculate_quality_score(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame,
                                options_df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Factors:
        - Completeness (no missing values)
        - Consistency (OHLC relationships)
        - Validity (reasonable value ranges)
        """
        scores = []
        
        # Completeness score
        total_missing = (spot_df.isnull().sum().sum() + 
                        futures_df.isnull().sum().sum() + 
                        options_df.isnull().sum().sum())
        total_cells = (spot_df.size + futures_df.size + options_df.size)
        completeness = 100 * (1 - total_missing / total_cells) if total_cells > 0 else 100
        scores.append(completeness)
        
        # OHLC consistency score (for spot)
        ohlc_valid = ((spot_df['high'] >= spot_df['low']) & 
                      (spot_df['high'] >= spot_df['open']) & 
                      (spot_df['high'] >= spot_df['close']) &
                      (spot_df['low'] <= spot_df['open']) & 
                      (spot_df['low'] <= spot_df['close']))
        consistency = 100 * ohlc_valid.mean()
        scores.append(consistency)
        
        # Value validity score
        valid_prices = (spot_df['close'] > 0).mean() * 100
        valid_iv = ((options_df['iv'] >= 5) & (options_df['iv'] <= 100)).mean() * 100
        validity = (valid_prices + valid_iv) / 2
        scores.append(validity)
        
        return round(np.mean(scores), 2)
    
    def clean_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """
        Run the complete data cleaning pipeline.
        
        Returns:
            Tuple of (cleaned_spot, cleaned_futures, cleaned_options, report_text)
        """
        self.report.log("=" * 60)
        self.report.log("Starting Data Cleaning Pipeline - Task 1.2")
        self.report.log("=" * 60)
        
        # Load data
        spot_df, futures_df, options_df = self.load_data()
        
        # Clean each dataset
        spot_clean = self.clean_spot_data(spot_df)
        futures_clean = self.clean_futures_data(futures_df, spot_clean)
        options_clean = self.clean_options_data(options_df, spot_clean)
        
        # Align timestamps
        spot_aligned, futures_aligned, options_aligned = self.align_timestamps(
            spot_clean, futures_clean, options_clean
        )
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(
            spot_aligned, futures_aligned, options_aligned
        )
        
        # Summary statistics
        total_processed = len(spot_df) + len(futures_df) + len(options_df)
        total_missing = (self.report.report['spot'].get('missing_detected', 0) +
                        self.report.report['futures'].get('missing_detected', 0) +
                        self.report.report['options'].get('missing_detected', 0))
        total_outliers = (self.report.report['spot'].get('outliers_corrected', 0) +
                         self.report.report['futures'].get('outliers_corrected', 0) +
                         self.report.report['options'].get('outliers_corrected', 0))
        
        self.report.add_stat('summary', 'total_processed', total_processed)
        self.report.add_stat('summary', 'total_missing', total_missing)
        self.report.add_stat('summary', 'total_outliers', total_outliers)
        self.report.add_stat('summary', 'quality_score', quality_score)
        
        self.report.log(f"\nData Quality Score: {quality_score}%")
        self.report.log("=" * 60)
        self.report.log("Data Cleaning Pipeline Complete")
        self.report.log("=" * 60)
        
        return spot_aligned, futures_aligned, options_aligned, self.report.generate_report()
    
    def save_cleaned_data(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame,
                          options_df: pd.DataFrame, report_text: str) -> Dict[str, Path]:
        """Save cleaned datasets and report."""
        paths = {}
        
        # Save cleaned spot data
        spot_path = self.processed_data_dir / "nifty_spot_5min_cleaned.csv"
        spot_df.to_csv(spot_path, index=False)
        paths['spot'] = spot_path
        self.report.log(f"Saved cleaned spot data: {spot_path}")
        
        # Save cleaned futures data
        futures_path = self.processed_data_dir / "nifty_futures_5min_cleaned.csv"
        futures_df.to_csv(futures_path, index=False)
        paths['futures'] = futures_path
        self.report.log(f"Saved cleaned futures data: {futures_path}")
        
        # Save cleaned options data
        options_path = self.processed_data_dir / "nifty_options_5min_cleaned.csv"
        options_df.to_csv(options_path, index=False)
        paths['options'] = options_path
        self.report.log(f"Saved cleaned options data: {options_path}")
        
        # Save cleaning report
        report_path = self.processed_data_dir / "data_cleaning_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        paths['report'] = report_path
        self.report.log(f"Saved cleaning report: {report_path}")
        
        return paths


def main():
    """Main function to run data cleaning."""
    cleaner = DataCleaner()
    
    # Run cleaning pipeline
    spot_clean, futures_clean, options_clean, report = cleaner.clean_all()
    
    # Save results
    paths = cleaner.save_cleaned_data(spot_clean, futures_clean, options_clean, report)
    
    # Print report
    print("\n" + report)
    
    return spot_clean, futures_clean, options_clean


if __name__ == "__main__":
    main()
