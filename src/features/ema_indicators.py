"""
EMA Indicators Module - Task 2.1

This module creates Exponential Moving Average (EMA) indicators
for trading signal generation.

EMAs:
- EMA(5)  - Fast moving average (short-term trend)
- EMA(15) - Slow moving average (medium-term trend)
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


class EMAIndicators:
    """
    Exponential Moving Average (EMA) calculator.
    
    EMA gives more weight to recent prices, making it more responsive
    to new information compared to Simple Moving Average (SMA).
    
    Formula:
        EMA_today = Price_today × k + EMA_yesterday × (1 - k)
        where k = 2 / (period + 1)
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 15):
        """
        Initialize EMA calculator.
        
        Args:
            fast_period: Period for fast EMA (default: 5)
            slow_period: Period for slow EMA (default: 15)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Calculate smoothing factors
        self.fast_alpha = 2 / (fast_period + 1)
        self.slow_alpha = 2 / (slow_period + 1)
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate EMA for a price series.
        
        Args:
            prices: Series of prices (typically close prices)
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_fast_ema(self, prices: pd.Series) -> pd.Series:
        """Calculate fast EMA (5-period)."""
        return self.calculate_ema(prices, self.fast_period)
    
    def calculate_slow_ema(self, prices: pd.Series) -> pd.Series:
        """Calculate slow EMA (15-period)."""
        return self.calculate_ema(prices, self.slow_period)
    
    def add_ema_features(self, df: pd.DataFrame, price_col: str = 'spot_close') -> pd.DataFrame:
        """
        Add EMA features to a DataFrame.
        
        Adds:
        - ema_fast: Fast EMA (5-period)
        - ema_slow: Slow EMA (15-period)
        - ema_diff: Difference between fast and slow EMA
        - ema_diff_pct: Percentage difference
        - ema_crossover: Signal when fast crosses slow (1=bullish, -1=bearish, 0=none)
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices (default: 'spot_close')
            
        Returns:
            DataFrame with EMA features added
        """
        df = df.copy()
        
        # Calculate EMAs
        df['ema_fast'] = self.calculate_fast_ema(df[price_col])
        df['ema_slow'] = self.calculate_slow_ema(df[price_col])
        
        # EMA difference (fast - slow)
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        
        # Percentage difference
        df['ema_diff_pct'] = (df['ema_diff'] / df['ema_slow']) * 100
        
        # Crossover detection
        df['ema_crossover'] = self._detect_crossover(df['ema_fast'], df['ema_slow'])
        
        # Trend direction based on EMA relationship
        df['ema_trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        return df
    
    def _detect_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> pd.Series:
        """
        Detect EMA crossovers.
        
        Returns:
            1  = Bullish crossover (fast crosses above slow)
            -1 = Bearish crossover (fast crosses below slow)
            0  = No crossover
        """
        # Current position: fast > slow
        position = (fast_ema > slow_ema).astype(int)
        
        # Previous position
        prev_position = position.shift(1)
        
        # Crossover occurs when position changes
        crossover = position - prev_position
        
        return crossover.fillna(0).astype(int)
    
    def get_ema_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for EMA features.
        
        Args:
            df: DataFrame with EMA features
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'fast_alpha': round(self.fast_alpha, 4),
            'slow_alpha': round(self.slow_alpha, 4),
            'total_records': len(df),
            'bullish_crossovers': (df['ema_crossover'] == 1).sum(),
            'bearish_crossovers': (df['ema_crossover'] == -1).sum(),
            'avg_ema_diff': round(df['ema_diff'].mean(), 2),
            'avg_ema_diff_pct': round(df['ema_diff_pct'].mean(), 4),
            'time_in_uptrend_pct': round((df['ema_trend'] == 1).mean() * 100, 2),
            'time_in_downtrend_pct': round((df['ema_trend'] == -1).mean() * 100, 2),
        }
        
        return summary


def add_ema_to_dataset(input_path: Path, output_path: Path = None) -> pd.DataFrame:
    """
    Add EMA indicators to a dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output (optional)
        
    Returns:
        DataFrame with EMA features
    """
    print("=" * 60)
    print("EMA Indicators - Task 2.1")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Records: {len(df)}")
    
    # Initialize EMA calculator
    ema = EMAIndicators(fast_period=5, slow_period=15)
    
    # Add EMA features
    print("\nCalculating EMA indicators...")
    print(f"  Fast EMA period: {ema.fast_period}")
    print(f"  Slow EMA period: {ema.slow_period}")
    
    df = ema.add_ema_features(df, price_col='spot_close')
    
    # Get summary
    summary = ema.get_ema_summary(df)
    
    print("\nEMA Summary:")
    print(f"  Bullish crossovers: {summary['bullish_crossovers']}")
    print(f"  Bearish crossovers: {summary['bearish_crossovers']}")
    print(f"  Time in uptrend: {summary['time_in_uptrend_pct']}%")
    print(f"  Time in downtrend: {summary['time_in_downtrend_pct']}%")
    print(f"  Avg EMA difference: {summary['avg_ema_diff']} points")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("EMA INDICATORS COMPLETE")
    print("=" * 60)
    
    return df, summary


def main():
    """Main function to add EMA indicators."""
    from ..data.config import PROCESSED_DATA_DIR
    
    input_path = PROCESSED_DATA_DIR / "nifty_merged_5min.csv"
    output_path = PROCESSED_DATA_DIR / "nifty_features_5min.csv"
    
    df, summary = add_ema_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with EMA:")
    print(df[['timestamp', 'spot_close', 'ema_fast', 'ema_slow', 
              'ema_diff', 'ema_crossover', 'ema_trend']].head(20))
    
    return df


if __name__ == "__main__":
    main()
