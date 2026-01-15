"""
ML Problem Definition Module - Task 5.1

Binary Classification Problem:
- Predict if a trade signal will be profitable
- Target: 1 if trade is profitable, 0 otherwise

Features:
- All engineered features (EMA, Greeks, IV, derived)
- Regime features
- Time-based features
- Lag features
- Signal strength features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MLDatasetBuilder:
    """
    Builds ML dataset for trade profitability prediction.
    
    Problem: Binary classification
    Target: 1 = profitable trade, 0 = unprofitable trade
    
    Features include:
    1. Engineered features (EMA, Greeks, IV, PCR, etc.)
    2. Regime features (current regime, regime probabilities)
    3. Time-based features (hour, day of week, time since open)
    4. Lag features (past values of key indicators)
    5. Signal strength features (EMA difference magnitude, crossover strength)
    """
    
    # Base features from existing data
    BASE_FEATURES = [
        # EMA features
        'ema_fast', 'ema_slow', 'ema_diff', 'ema_diff_pct', 'ema_trend',
        
        # Spot features
        'spot_close', 'spot_volume',
        
        # Futures features
        'fut_close', 'fut_oi', 'fut_basis', 'fut_basis_pct', 'fut_dte',
        
        # IV features
        'avg_iv', 'iv_spread', 'iv_percentile',
        
        # PCR features
        'pcr_oi', 'pcr_volume',
        
        # Greeks features
        'greeks_ce_atm_delta', 'greeks_ce_atm_gamma', 'greeks_ce_atm_theta',
        'greeks_ce_atm_vega', 'greeks_ce_atm_rho',
        'greeks_pe_atm_delta', 'greeks_pe_atm_gamma', 'greeks_pe_atm_theta',
        'greeks_pe_atm_vega', 'greeks_pe_atm_rho',
        
        # Derived features
        'futures_basis_pct', 'delta_neutral_ratio', 'gex_net', 'price_momentum',
        
        # Returns and volatility
        'spot_return_1', 'spot_return_5', 'spot_volatility_1h',
        
        # Regime features
        'regime_filled', 'regime_prob_up', 'regime_prob_down', 'regime_prob_side',
    ]
    
    # Time features to create
    TIME_FEATURES = [
        'hour', 'minute', 'day_of_week',
        'time_since_open',  # Minutes since market open
        'time_to_close',    # Minutes until market close
        'is_first_hour',    # First hour of trading
        'is_last_hour',     # Last hour of trading
    ]
    
    # Lag periods for lag features
    LAG_PERIODS = [1, 3, 6, 12]  # 5min, 15min, 30min, 1hour
    
    def __init__(self, train_ratio: float = 0.7):
        """
        Initialize dataset builder.
        
        Args:
            train_ratio: Proportion for training (default: 70%)
        """
        self.train_ratio = train_ratio
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_target_variable(self, df: pd.DataFrame, 
                                trade_log_path: Path = None) -> pd.DataFrame:
        """
        Create target variable: 1 if trade is profitable, 0 otherwise.
        
        Target is created at signal points (where signal_type is LONG_ENTRY or SHORT_ENTRY).
        """
        df = df.copy()
        
        # Initialize target column
        df['target'] = np.nan
        df['trade_pnl'] = np.nan
        
        # Load trade log if available
        if trade_log_path and trade_log_path.exists():
            trade_log = pd.read_csv(trade_log_path)
            trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
            
            # Map trades to signals
            for _, trade in trade_log.iterrows():
                entry_time = trade['entry_time']
                pnl = trade['pnl']
                
                # Find the signal row (one bar before entry)
                signal_mask = df['timestamp'] == entry_time - pd.Timedelta(minutes=5)
                
                if signal_mask.any():
                    df.loc[signal_mask, 'target'] = 1 if pnl > 0 else 0
                    df.loc[signal_mask, 'trade_pnl'] = pnl
        else:
            # Create target from signal and future returns
            # For each entry signal, look at future price movement
            entry_signals = df['signal_type'].isin(['LONG_ENTRY', 'SHORT_ENTRY'])
            
            for idx in df[entry_signals].index:
                signal_type = df.loc[idx, 'signal_type']
                entry_price = df.loc[idx, 'spot_close']
                
                # Look ahead for exit (next opposite crossover)
                future_df = df.loc[idx+1:]
                
                if signal_type == 'LONG_ENTRY':
                    # Exit on bearish crossover
                    exit_mask = future_df['ema_crossover'] == -1
                else:
                    # Exit on bullish crossover
                    exit_mask = future_df['ema_crossover'] == 1
                
                if exit_mask.any():
                    exit_idx = future_df[exit_mask].index[0]
                    exit_price = df.loc[exit_idx, 'spot_close']
                    
                    if signal_type == 'LONG_ENTRY':
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    
                    df.loc[idx, 'target'] = 1 if pnl > 0 else 0
                    df.loc[idx, 'trade_pnl'] = pnl
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Time since market open (9:15 AM)
        market_open_minutes = 9 * 60 + 15
        current_minutes = df['hour'] * 60 + df['minute']
        df['time_since_open'] = current_minutes - market_open_minutes
        
        # Time to market close (3:30 PM)
        market_close_minutes = 15 * 60 + 30
        df['time_to_close'] = market_close_minutes - current_minutes
        
        # Session indicators
        df['is_first_hour'] = (df['hour'] == 9) | ((df['hour'] == 10) & (df['minute'] < 15))
        df['is_last_hour'] = (df['hour'] == 14) | ((df['hour'] == 15) & (df['minute'] <= 30))
        
        # Convert boolean to int
        df['is_first_hour'] = df['is_first_hour'].astype(int)
        df['is_last_hour'] = df['is_last_hour'].astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for key indicators.
        """
        df = df.copy()
        
        # Features to create lags for
        lag_cols = [
            'spot_return_1', 'ema_diff', 'avg_iv', 'pcr_oi',
            'greeks_ce_atm_delta', 'gex_net', 'spot_volatility_1h'
        ]
        
        for col in lag_cols:
            if col in df.columns:
                for lag in self.LAG_PERIODS:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_signal_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create signal strength features.
        """
        df = df.copy()
        
        # EMA crossover strength (magnitude of difference at crossover)
        df['crossover_strength'] = np.abs(df['ema_diff'])
        
        # EMA momentum (rate of change of EMA difference)
        df['ema_diff_momentum'] = df['ema_diff'] - df['ema_diff'].shift(1)
        
        # Trend strength (how long in current trend)
        df['trend_duration'] = df.groupby((df['ema_trend'] != df['ema_trend'].shift()).cumsum()).cumcount() + 1
        
        # Regime confidence (max of regime probabilities)
        if all(col in df.columns for col in ['regime_prob_up', 'regime_prob_down', 'regime_prob_side']):
            df['regime_confidence'] = df[['regime_prob_up', 'regime_prob_down', 'regime_prob_side']].max(axis=1)
        
        # IV rank (percentile of current IV)
        if 'avg_iv' in df.columns:
            df['iv_rank'] = df['avg_iv'].rolling(window=75).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 
                if x.max() != x.min() else 50
            )
        
        # Volume ratio (current vs average)
        if 'spot_volume' in df.columns:
            df['volume_ratio'] = df['spot_volume'] / df['spot_volume'].rolling(window=75).mean()
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features.
        """
        df = df.copy()
        
        # Rolling statistics for returns
        if 'spot_return_1' in df.columns:
            df['return_mean_12'] = df['spot_return_1'].rolling(window=12).mean()
            df['return_std_12'] = df['spot_return_1'].rolling(window=12).std()
            df['return_skew_12'] = df['spot_return_1'].rolling(window=12).skew()
        
        # Rolling statistics for IV
        if 'avg_iv' in df.columns:
            df['iv_mean_12'] = df['avg_iv'].rolling(window=12).mean()
            df['iv_change'] = df['avg_iv'] - df['avg_iv'].shift(1)
        
        # Rolling statistics for PCR
        if 'pcr_oi' in df.columns:
            df['pcr_mean_12'] = df['pcr_oi'].rolling(window=12).mean()
            df['pcr_change'] = df['pcr_oi'] - df['pcr_oi'].shift(1)
        
        return df
    
    def build_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build complete feature matrix for ML.
        
        Returns:
            Tuple of (feature DataFrame, list of feature names)
        """
        print("Building feature matrix...")
        
        # Create all features
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_signal_strength_features(df)
        df = self.create_rolling_features(df)
        
        # Collect all feature columns
        feature_cols = []
        
        # Add base features
        for col in self.BASE_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add time features
        for col in ['hour', 'minute', 'day_of_week', 'time_since_open', 'time_to_close',
                    'is_first_hour', 'is_last_hour']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add lag features
        for col in df.columns:
            if '_lag' in col:
                feature_cols.append(col)
        
        # Add signal strength features
        for col in ['crossover_strength', 'ema_diff_momentum', 'trend_duration',
                    'regime_confidence', 'iv_rank', 'volume_ratio']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add rolling features
        for col in ['return_mean_12', 'return_std_12', 'return_skew_12',
                    'iv_mean_12', 'iv_change', 'pcr_mean_12', 'pcr_change']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        self.feature_names = feature_cols
        print(f"  Total features: {len(feature_cols)}")
        
        return df, feature_cols
    
    def prepare_ml_dataset(self, df: pd.DataFrame, 
                           trade_log_path: Path = None) -> Dict:
        """
        Prepare complete ML dataset with train/test split.
        
        Args:
            df: Input DataFrame
            trade_log_path: Path to trade log CSV
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, feature_names
        """
        print("=" * 60)
        print("ML DATASET PREPARATION - Task 5.1")
        print("=" * 60)
        
        # Create target variable
        print("\nCreating target variable...")
        df = self.create_target_variable(df, trade_log_path)
        
        # Build feature matrix
        df, feature_cols = self.build_feature_matrix(df)
        
        # Filter to rows with valid target
        ml_df = df[df['target'].notna()].copy()
        print(f"\nSamples with target: {len(ml_df)}")
        
        # Check target distribution
        target_dist = ml_df['target'].value_counts()
        print(f"Target distribution:")
        print(f"  Profitable (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(ml_df)*100:.1f}%)")
        print(f"  Unprofitable (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(ml_df)*100:.1f}%)")
        
        # Prepare features and target
        X = ml_df[feature_cols].copy()
        y = ml_df['target'].copy()
        
        # Handle missing values
        print(f"\nHandling missing values...")
        missing_before = X.isnull().sum().sum()
        X = X.ffill().bfill()
        X = X.fillna(0)  # Fill any remaining NaN with 0
        missing_after = X.isnull().sum().sum()
        print(f"  Missing values: {missing_before} → {missing_after}")
        
        # Split data (time-based split to avoid lookahead bias)
        n_train = int(len(X) * self.train_ratio)
        
        X_train = X.iloc[:n_train]
        X_test = X.iloc[n_train:]
        y_train = y.iloc[:n_train]
        y_test = y.iloc[n_train:]
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)} ({self.train_ratio*100:.0f}%)")
        print(f"  Testing samples: {len(X_test)} ({(1-self.train_ratio)*100:.0f}%)")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index
        )
        
        # Prepare result
        result = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_unscaled': X_train,
            'X_test_unscaled': X_test,
            'feature_names': feature_cols,
            'scaler': self.scaler,
            'ml_df': ml_df,
            'target_distribution': {
                'train': y_train.value_counts().to_dict(),
                'test': y_test.value_counts().to_dict()
            }
        }
        
        print("\n" + "=" * 60)
        print("ML DATASET READY")
        print("=" * 60)
        
        return result
    
    def get_feature_summary(self, feature_names: List[str]) -> Dict:
        """Get summary of features by category."""
        summary = {
            'total_features': len(feature_names),
            'categories': {
                'EMA': [f for f in feature_names if 'ema' in f.lower()],
                'Greeks': [f for f in feature_names if 'greek' in f.lower()],
                'IV': [f for f in feature_names if 'iv' in f.lower()],
                'PCR': [f for f in feature_names if 'pcr' in f.lower()],
                'Regime': [f for f in feature_names if 'regime' in f.lower()],
                'Time': [f for f in feature_names if any(t in f for t in ['hour', 'minute', 'day', 'time', 'first', 'last'])],
                'Lag': [f for f in feature_names if 'lag' in f.lower()],
                'Signal': [f for f in feature_names if any(s in f for s in ['crossover', 'trend', 'strength', 'momentum'])],
                'Rolling': [f for f in feature_names if any(r in f for r in ['mean_12', 'std_12', 'skew', 'change'])],
            }
        }
        
        # Count by category
        summary['counts'] = {k: len(v) for k, v in summary['categories'].items()}
        
        return summary


def prepare_ml_dataset(input_path: Path, trade_log_path: Path = None) -> Dict:
    """
    Main function to prepare ML dataset.
    
    Args:
        input_path: Path to features CSV
        trade_log_path: Path to trade log CSV
        
    Returns:
        Dictionary with ML dataset components
    """
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    
    # Build dataset
    builder = MLDatasetBuilder(train_ratio=0.7)
    result = builder.prepare_ml_dataset(df, trade_log_path)
    
    # Print feature summary
    summary = builder.get_feature_summary(result['feature_names'])
    print("\nFeature Summary:")
    for category, count in summary['counts'].items():
        if count > 0:
            print(f"  {category}: {count} features")
    
    return result


def main():
    """Main function."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    trade_log_path = Path("data/processed/trade_log.csv")
    
    result = prepare_ml_dataset(input_path, trade_log_path)
    
    # Print sample
    print("\nSample features (first 5 rows):")
    print(result['X_train'].head())
    
    return result


if __name__ == "__main__":
    main()
