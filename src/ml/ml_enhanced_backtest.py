"""
ML-Enhanced Backtest Module - Task 5.3

Backtest strategy with ML filter:
- Only take trades when ML model predicts profitable (confidence > 0.5)
- Compare: Baseline vs XGBoost vs LSTM
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings

from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Import strategy components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.ema_regime_strategy import EMARegimeStrategy, Trade, Position
from src.strategy.backtester import PerformanceMetrics


class MLEnhancedStrategy(EMARegimeStrategy):
    """
    EMA Regime Strategy enhanced with ML prediction filter.
    
    Only takes trades when ML model predicts the trade will be profitable
    with confidence > threshold.
    """
    
    def __init__(self, ml_model=None, confidence_threshold: float = 0.5,
                 fast_period: int = 5, slow_period: int = 15):
        """
        Initialize ML-enhanced strategy.
        
        Args:
            ml_model: Trained ML model with predict_proba method
            confidence_threshold: Minimum confidence to take trade (default: 0.5)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
        """
        super().__init__(fast_period, slow_period)
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        self.filtered_signals = 0
        self.total_signals = 0
    
    def should_take_trade(self, features: pd.Series) -> Tuple[bool, float]:
        """
        Check if ML model predicts trade will be profitable.
        
        Args:
            features: Feature values for the signal
            
        Returns:
            Tuple of (should_take, confidence)
        """
        if self.ml_model is None:
            return True, 1.0
        
        try:
            # Reshape for prediction
            X = features.values.reshape(1, -1)
            
            # Get probability of profitable trade
            proba = self.ml_model.predict_proba(X)[0, 1]
            
            return proba >= self.confidence_threshold, proba
        except Exception as e:
            # If prediction fails, take the trade
            return True, 0.5
    
    def backtest_with_ml_filter(self, df: pd.DataFrame, 
                                 feature_cols: List[str]) -> Tuple[pd.DataFrame, List[Trade]]:
        """
        Run backtest with ML filter applied to signals.
        
        Args:
            df: DataFrame with price, signal, and feature data
            feature_cols: List of feature column names for ML model
            
        Returns:
            Tuple of (DataFrame with positions, list of trades)
        """
        df = df.copy()
        
        # Generate base signals
        df = self.generate_signals(df)
        
        # Reset state
        self.position = Position.FLAT
        self.current_trade = None
        self.trades = []
        self.filtered_signals = 0
        self.total_signals = 0
        
        # Track position for each bar
        positions = []
        trade_ids = []
        ml_confidence = []
        
        # Iterate through data
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Get next candle open price
            if i + 1 < len(df):
                next_open = df.iloc[i + 1]['spot_open']
                next_time = df.iloc[i + 1]['timestamp']
            else:
                next_open = row['spot_close']
                next_time = row['timestamp']
            
            signal_type = row['signal_type']
            confidence = 0.5
            
            # Process signals with ML filter
            if self.position == Position.FLAT:
                if signal_type in ['LONG_ENTRY', 'SHORT_ENTRY']:
                    self.total_signals += 1
                    
                    # Check ML prediction
                    if self.ml_model is not None and all(col in df.columns for col in feature_cols):
                        features = row[feature_cols]
                        should_take, confidence = self.should_take_trade(features)
                    else:
                        should_take = True
                        confidence = 1.0
                    
                    if should_take:
                        if signal_type == 'LONG_ENTRY':
                            self.position = Position.LONG
                            self.current_trade = Trade(
                                entry_time=next_time,
                                entry_price=next_open,
                                entry_type='LONG',
                                regime_at_entry=row.get('regime_filled', 0)
                            )
                        else:
                            self.position = Position.SHORT
                            self.current_trade = Trade(
                                entry_time=next_time,
                                entry_price=next_open,
                                entry_type='SHORT',
                                regime_at_entry=row.get('regime_filled', 0)
                            )
                    else:
                        self.filtered_signals += 1
            
            elif self.position == Position.LONG:
                if signal_type in ['LONG_EXIT', 'SHORT_ENTRY']:
                    # Exit LONG
                    if self.current_trade:
                        duration = i
                        self.current_trade.close(next_time, next_open, duration)
                        self.trades.append(self.current_trade)
                    
                    self.position = Position.FLAT
                    self.current_trade = None
                    
                    # Check for SHORT entry
                    if signal_type == 'SHORT_ENTRY':
                        self.total_signals += 1
                        
                        if self.ml_model is not None and all(col in df.columns for col in feature_cols):
                            features = row[feature_cols]
                            should_take, confidence = self.should_take_trade(features)
                        else:
                            should_take = True
                            confidence = 1.0
                        
                        if should_take:
                            self.position = Position.SHORT
                            self.current_trade = Trade(
                                entry_time=next_time,
                                entry_price=next_open,
                                entry_type='SHORT',
                                regime_at_entry=row.get('regime_filled', 0)
                            )
                        else:
                            self.filtered_signals += 1
            
            elif self.position == Position.SHORT:
                if signal_type in ['SHORT_EXIT', 'LONG_ENTRY']:
                    # Exit SHORT
                    if self.current_trade:
                        duration = i
                        self.current_trade.close(next_time, next_open, duration)
                        self.trades.append(self.current_trade)
                    
                    self.position = Position.FLAT
                    self.current_trade = None
                    
                    # Check for LONG entry
                    if signal_type == 'LONG_ENTRY':
                        self.total_signals += 1
                        
                        if self.ml_model is not None and all(col in df.columns for col in feature_cols):
                            features = row[feature_cols]
                            should_take, confidence = self.should_take_trade(features)
                        else:
                            should_take = True
                            confidence = 1.0
                        
                        if should_take:
                            self.position = Position.LONG
                            self.current_trade = Trade(
                                entry_time=next_time,
                                entry_price=next_open,
                                entry_type='LONG',
                                regime_at_entry=row.get('regime_filled', 0)
                            )
                        else:
                            self.filtered_signals += 1
            
            positions.append(self.position.value)
            trade_ids.append(len(self.trades))
            ml_confidence.append(confidence)
        
        # Close any open trade
        if self.current_trade:
            last_row = df.iloc[-1]
            self.current_trade.close(last_row['timestamp'], last_row['spot_close'], len(df)-1)
            self.trades.append(self.current_trade)
        
        df['position'] = positions
        df['trade_id'] = trade_ids
        df['ml_confidence'] = ml_confidence
        
        return df, self.trades


def run_ml_enhanced_backtest(data_path: Path, ml_dataset_path: Path,
                              xgb_model_path: Path, lstm_model_path: Path = None,
                              confidence_threshold: float = 0.3) -> Dict:
    """
    Run backtests comparing Baseline vs XGBoost vs LSTM.
    
    Args:
        data_path: Path to features CSV
        ml_dataset_path: Path to ML dataset pickle
        xgb_model_path: Path to XGBoost model
        lstm_model_path: Path to LSTM model (optional)
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print("ML-ENHANCED BACKTEST - Task 5.3")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    
    # Load ML dataset for feature names
    print(f"Loading ML dataset from: {ml_dataset_path}")
    with open(ml_dataset_path, 'rb') as f:
        ml_data = pickle.load(f)
    feature_cols = ml_data['feature_names']
    print(f"  Features: {len(feature_cols)}")
    
    # Load XGBoost model
    print(f"Loading XGBoost model from: {xgb_model_path}")
    with open(xgb_model_path, 'rb') as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data['model']
    
    # Load LSTM model if available
    lstm_model = None
    if lstm_model_path:
        try:
            from tensorflow import keras
            # Try .keras extension first
            lstm_keras_path = Path(str(lstm_model_path) + '.keras')
            if lstm_keras_path.exists():
                lstm_model = keras.models.load_model(lstm_keras_path)
                print(f"Loaded LSTM model from: {lstm_keras_path}")
            elif lstm_model_path.with_suffix('.keras').exists():
                lstm_model = keras.models.load_model(lstm_model_path.with_suffix('.keras'))
                print(f"Loaded LSTM model from: {lstm_model_path.with_suffix('.keras')}")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
    
    # Initialize metrics calculator
    metrics_calc = PerformanceMetrics()
    
    results = {}
    
    # =========================================
    # 1. Baseline (No ML Filter)
    # =========================================
    print("\n" + "=" * 50)
    print("1. BASELINE (No ML Filter)")
    print("=" * 50)
    
    baseline_strategy = MLEnhancedStrategy(ml_model=None)
    df_baseline, baseline_trades = baseline_strategy.backtest_with_ml_filter(df, feature_cols)
    
    # Build equity curve
    equity_baseline = build_equity_curve(df_baseline, baseline_trades)
    baseline_metrics = metrics_calc.calculate_all_metrics(equity_baseline, baseline_trades)
    
    print(f"\nBaseline Results:")
    print(f"  Total Signals: {baseline_strategy.total_signals}")
    print(f"  Trades Taken: {len(baseline_trades)}")
    print(f"  Win Rate: {baseline_metrics['win_rate_pct']:.2f}%")
    print(f"  Total Return: {baseline_metrics['total_return_pct']:.4f}%")
    print(f"  Profit Factor: {baseline_metrics['profit_factor']:.4f}")
    
    results['baseline'] = {
        'trades': len(baseline_trades),
        'filtered': 0,
        'metrics': baseline_metrics
    }
    
    # =========================================
    # 2. XGBoost Filter (confidence > 0.5)
    # =========================================
    print("\n" + "=" * 50)
    print(f"2. XGBOOST FILTER (confidence > {confidence_threshold})")
    print("=" * 50)
    
    # Prepare features for ML prediction
    df_ml = prepare_ml_features(df, feature_cols, ml_data['scaler'])
    
    xgb_strategy = MLEnhancedStrategy(ml_model=xgb_model, confidence_threshold=confidence_threshold)
    df_xgb, xgb_trades = xgb_strategy.backtest_with_ml_filter(df_ml, feature_cols)
    
    equity_xgb = build_equity_curve(df_xgb, xgb_trades)
    xgb_metrics = metrics_calc.calculate_all_metrics(equity_xgb, xgb_trades)
    
    print(f"\nXGBoost Results:")
    print(f"  Total Signals: {xgb_strategy.total_signals}")
    print(f"  Signals Filtered: {xgb_strategy.filtered_signals}")
    print(f"  Trades Taken: {len(xgb_trades)}")
    print(f"  Win Rate: {xgb_metrics['win_rate_pct']:.2f}%")
    print(f"  Total Return: {xgb_metrics['total_return_pct']:.4f}%")
    print(f"  Profit Factor: {xgb_metrics['profit_factor']:.4f}")
    
    results['xgboost'] = {
        'trades': len(xgb_trades),
        'filtered': xgb_strategy.filtered_signals,
        'metrics': xgb_metrics
    }
    
    # =========================================
    # 3. LSTM Filter (if available)
    # =========================================
    if lstm_model is not None:
        print("\n" + "=" * 50)
        print(f"3. LSTM FILTER (confidence > {confidence_threshold})")
        print("=" * 50)
        
        # LSTM needs sequence data - use wrapper
        lstm_wrapper = LSTMWrapper(lstm_model, sequence_length=10)
        
        lstm_strategy = MLEnhancedStrategy(ml_model=lstm_wrapper, confidence_threshold=confidence_threshold)
        df_lstm, lstm_trades = lstm_strategy.backtest_with_ml_filter(df_ml, feature_cols)
        
        equity_lstm = build_equity_curve(df_lstm, lstm_trades)
        lstm_metrics = metrics_calc.calculate_all_metrics(equity_lstm, lstm_trades)
        
        print(f"\nLSTM Results:")
        print(f"  Total Signals: {lstm_strategy.total_signals}")
        print(f"  Signals Filtered: {lstm_strategy.filtered_signals}")
        print(f"  Trades Taken: {len(lstm_trades)}")
        print(f"  Win Rate: {lstm_metrics['win_rate_pct']:.2f}%")
        print(f"  Total Return: {lstm_metrics['total_return_pct']:.4f}%")
        print(f"  Profit Factor: {lstm_metrics['profit_factor']:.4f}")
        
        results['lstm'] = {
            'trades': len(lstm_trades),
            'filtered': lstm_strategy.filtered_signals,
            'metrics': lstm_metrics
        }
    else:
        print("\n" + "=" * 50)
        print("3. LSTM FILTER - SKIPPED (model not available)")
        print("=" * 50)
        results['lstm'] = {'error': 'Model not available'}
    
    # =========================================
    # Comparison Summary
    # =========================================
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'XGBoost':<15} {'LSTM':<15}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('Total Trades', 'trades'),
        ('Filtered Signals', 'filtered'),
        ('Win Rate %', 'win_rate_pct'),
        ('Total Return %', 'total_return_pct'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown %', 'max_drawdown_pct'),
        ('Profit Factor', 'profit_factor'),
    ]
    
    for display_name, metric_key in metrics_to_compare:
        baseline_val = get_metric_value(results['baseline'], metric_key)
        xgb_val = get_metric_value(results['xgboost'], metric_key)
        lstm_val = get_metric_value(results.get('lstm', {}), metric_key)
        
        print(f"{display_name:<20} {baseline_val:<15} {xgb_val:<15} {lstm_val:<15}")
    
    print("\n" + "=" * 70)
    print("ML-ENHANCED BACKTEST COMPLETE")
    print("=" * 70)
    
    return results


def prepare_ml_features(df: pd.DataFrame, feature_cols: List[str], 
                        scaler) -> pd.DataFrame:
    """Prepare features for ML prediction."""
    df = df.copy()
    
    # Forward fill regime
    if 'regime' in df.columns:
        df['regime_filled'] = df['regime'].ffill()
    
    # Create any missing features with defaults
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Fill missing values
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    
    # Scale features
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df


def build_equity_curve(df: pd.DataFrame, trades: List[Trade], 
                       initial_capital: float = 100000) -> pd.Series:
    """Build equity curve from trades."""
    equity = pd.Series(index=df['timestamp'], dtype=float)
    equity.iloc[0] = initial_capital
    
    cumulative_pnl = 0
    trade_pnl = {}
    
    for t in trades:
        if t.exit_time and t.pnl:
            trade_pnl[t.exit_time] = t.pnl
    
    for i, ts in enumerate(df['timestamp']):
        if i == 0:
            continue
        if ts in trade_pnl:
            cumulative_pnl += trade_pnl[ts]
        equity.iloc[i] = initial_capital + cumulative_pnl
    
    return equity.ffill()


def get_metric_value(result: Dict, metric_key: str) -> str:
    """Get formatted metric value from result dict."""
    if 'error' in result:
        return 'N/A'
    
    if metric_key in ['trades', 'filtered']:
        return str(result.get(metric_key, 0))
    
    metrics = result.get('metrics', {})
    value = metrics.get(metric_key, 0)
    
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


class LSTMWrapper:
    """Wrapper to make LSTM model compatible with sklearn-style predict_proba."""
    
    def __init__(self, model, sequence_length: int = 10):
        self.model = model
        self.sequence_length = sequence_length
        self.history = []
    
    def predict_proba(self, X):
        """Predict probability using LSTM with sequence history."""
        # Add to history
        self.history.append(X[0])
        
        # Keep only last sequence_length samples
        if len(self.history) > self.sequence_length:
            self.history = self.history[-self.sequence_length:]
        
        # If not enough history, return 0.5
        if len(self.history) < self.sequence_length:
            return np.array([[0.5, 0.5]])
        
        # Create sequence
        X_seq = np.array(self.history).reshape(1, self.sequence_length, -1)
        
        # Predict
        proba = self.model.predict(X_seq, verbose=0)[0, 0]
        
        return np.array([[1 - proba, proba]])


def main():
    """Main function."""
    data_path = Path("data/processed/nifty_features_5min.csv")
    ml_dataset_path = Path("data/processed/ml_dataset.pkl")
    xgb_model_path = Path("data/processed/xgboost_model.pkl")
    lstm_model_path = Path("data/processed/lstm_model")
    
    results = run_ml_enhanced_backtest(
        data_path, ml_dataset_path, xgb_model_path, lstm_model_path
    )
    
    # Save results
    output_path = Path("data/processed/ml_backtest_comparison.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
