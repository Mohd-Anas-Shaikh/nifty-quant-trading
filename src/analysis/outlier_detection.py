"""
Outlier Detection Module - Task 6.1

Identify profitable trades beyond 3-sigma (Z-score > 3).
Analyze features: regime, IV, ATR, time of day, Greeks, trade duration, EMA gap, PCR.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class OutlierDetector:
    """
    Detect and analyze outlier trades (high-performance trades).
    
    Outliers are defined as trades with Z-score > 3 (beyond 3 standard deviations).
    These represent exceptional trading opportunities.
    """
    
    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize outlier detector.
        
        Args:
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
        """
        self.z_threshold = z_threshold
        self.outliers = None
        self.analysis_results = {}
    
    def calculate_z_scores(self, pnl_series: pd.Series) -> pd.Series:
        """
        Calculate Z-scores for P&L values.
        
        Z-score = (value - mean) / std
        
        Args:
            pnl_series: Series of P&L values
            
        Returns:
            Series of Z-scores
        """
        mean_pnl = pnl_series.mean()
        std_pnl = pnl_series.std()
        
        if std_pnl == 0:
            return pd.Series(0, index=pnl_series.index)
        
        return (pnl_series - mean_pnl) / std_pnl
    
    def identify_outliers(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Identify trades with Z-score > threshold.
        
        Args:
            trade_log: DataFrame with trade data including 'pnl' column
            
        Returns:
            DataFrame of outlier trades
        """
        # Calculate Z-scores
        trade_log = trade_log.copy()
        trade_log['z_score'] = self.calculate_z_scores(trade_log['pnl'])
        
        # Identify positive outliers (exceptional profits)
        positive_outliers = trade_log[trade_log['z_score'] > self.z_threshold]
        
        # Also identify negative outliers for comparison
        negative_outliers = trade_log[trade_log['z_score'] < -self.z_threshold]
        
        self.outliers = {
            'positive': positive_outliers,
            'negative': negative_outliers,
            'all_trades': trade_log
        }
        
        return positive_outliers
    
    def merge_trade_features(self, trade_log: pd.DataFrame, 
                              features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge trade log with feature data at entry time.
        
        Args:
            trade_log: DataFrame with trade data
            features_df: DataFrame with all features
            
        Returns:
            Trade log with features merged
        """
        trade_log = trade_log.copy()
        features_df = features_df.copy()
        
        # Ensure timestamp columns are datetime
        trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Create a lookup for features by timestamp
        # We need to find features at entry time - 5 minutes (signal time)
        merged_trades = []
        
        for _, trade in trade_log.iterrows():
            entry_time = trade['entry_time']
            signal_time = entry_time - pd.Timedelta(minutes=5)
            
            # Find closest feature row
            time_diff = abs(features_df['timestamp'] - signal_time)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] <= pd.Timedelta(minutes=10):
                feature_row = features_df.loc[closest_idx]
                
                # Merge trade with features
                merged = trade.to_dict()
                
                # Add key features
                feature_cols = [
                    'regime', 'regime_filled', 'avg_iv', 'iv_spread', 'pcr_oi', 'pcr_volume',
                    'ema_fast', 'ema_slow', 'ema_diff', 'ema_diff_pct',
                    'greeks_ce_atm_delta', 'greeks_ce_atm_gamma', 'greeks_ce_atm_theta',
                    'greeks_ce_atm_vega', 'greeks_pe_atm_delta',
                    'spot_return_1', 'spot_volatility_1h', 'gex_net',
                    'futures_basis_pct', 'delta_neutral_ratio',
                    'hour', 'day_of_week'
                ]
                
                for col in feature_cols:
                    if col in feature_row.index:
                        merged[f'feat_{col}'] = feature_row[col]
                
                merged_trades.append(merged)
        
        return pd.DataFrame(merged_trades)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures volatility based on price range.
        """
        high = df['spot_high'] if 'spot_high' in df.columns else df['spot_close']
        low = df['spot_low'] if 'spot_low' in df.columns else df['spot_close']
        close = df['spot_close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def analyze_outlier_features(self, outliers: pd.DataFrame, 
                                  all_trades: pd.DataFrame) -> Dict:
        """
        Analyze features of outlier trades vs normal trades.
        
        Args:
            outliers: DataFrame of outlier trades
            all_trades: DataFrame of all trades
            
        Returns:
            Dictionary with analysis results
        """
        if len(outliers) == 0:
            return {'error': 'No outliers found'}
        
        # Normal trades (non-outliers)
        normal_trades = all_trades[all_trades['z_score'].abs() <= self.z_threshold]
        
        analysis = {
            'count': {
                'outliers': len(outliers),
                'normal': len(normal_trades),
                'total': len(all_trades)
            },
            'pnl_stats': {
                'outlier_mean': outliers['pnl'].mean(),
                'outlier_min': outliers['pnl'].min(),
                'outlier_max': outliers['pnl'].max(),
                'normal_mean': normal_trades['pnl'].mean() if len(normal_trades) > 0 else 0,
                'all_mean': all_trades['pnl'].mean(),
                'all_std': all_trades['pnl'].std()
            }
        }
        
        # Analyze each feature category
        feature_analysis = {}
        
        # 1. Regime Analysis
        if 'feat_regime_filled' in outliers.columns:
            regime_dist_outliers = outliers['feat_regime_filled'].value_counts(normalize=True)
            regime_dist_normal = normal_trades['feat_regime_filled'].value_counts(normalize=True) if len(normal_trades) > 0 else pd.Series()
            
            feature_analysis['regime'] = {
                'outlier_distribution': regime_dist_outliers.to_dict(),
                'normal_distribution': regime_dist_normal.to_dict() if len(regime_dist_normal) > 0 else {}
            }
        
        # 2. IV Analysis
        if 'feat_avg_iv' in outliers.columns:
            feature_analysis['iv'] = {
                'outlier_mean': outliers['feat_avg_iv'].mean(),
                'outlier_std': outliers['feat_avg_iv'].std(),
                'normal_mean': normal_trades['feat_avg_iv'].mean() if len(normal_trades) > 0 else 0,
                'normal_std': normal_trades['feat_avg_iv'].std() if len(normal_trades) > 0 else 0
            }
        
        # 3. Time of Day Analysis
        if 'feat_hour' in outliers.columns:
            hour_dist_outliers = outliers['feat_hour'].value_counts(normalize=True)
            hour_dist_normal = normal_trades['feat_hour'].value_counts(normalize=True) if len(normal_trades) > 0 else pd.Series()
            
            feature_analysis['time_of_day'] = {
                'outlier_distribution': hour_dist_outliers.to_dict(),
                'normal_distribution': hour_dist_normal.to_dict() if len(hour_dist_normal) > 0 else {},
                'outlier_peak_hour': outliers['feat_hour'].mode().iloc[0] if len(outliers) > 0 else None
            }
        
        # 4. Greeks Analysis
        greeks_cols = ['feat_greeks_ce_atm_delta', 'feat_greeks_ce_atm_gamma', 
                       'feat_greeks_ce_atm_theta', 'feat_greeks_ce_atm_vega']
        
        feature_analysis['greeks'] = {}
        for col in greeks_cols:
            if col in outliers.columns:
                greek_name = col.replace('feat_greeks_ce_atm_', '')
                feature_analysis['greeks'][greek_name] = {
                    'outlier_mean': outliers[col].mean(),
                    'normal_mean': normal_trades[col].mean() if len(normal_trades) > 0 else 0
                }
        
        # 5. Trade Duration Analysis
        if 'duration_bars' in outliers.columns:
            feature_analysis['duration'] = {
                'outlier_mean': outliers['duration_bars'].mean(),
                'outlier_median': outliers['duration_bars'].median(),
                'normal_mean': normal_trades['duration_bars'].mean() if len(normal_trades) > 0 else 0,
                'normal_median': normal_trades['duration_bars'].median() if len(normal_trades) > 0 else 0
            }
        
        # 6. EMA Gap Analysis
        if 'feat_ema_diff' in outliers.columns:
            feature_analysis['ema_gap'] = {
                'outlier_mean': outliers['feat_ema_diff'].mean(),
                'outlier_abs_mean': outliers['feat_ema_diff'].abs().mean(),
                'normal_mean': normal_trades['feat_ema_diff'].mean() if len(normal_trades) > 0 else 0,
                'normal_abs_mean': normal_trades['feat_ema_diff'].abs().mean() if len(normal_trades) > 0 else 0
            }
        
        # 7. PCR Analysis
        if 'feat_pcr_oi' in outliers.columns:
            feature_analysis['pcr'] = {
                'outlier_mean': outliers['feat_pcr_oi'].mean(),
                'outlier_std': outliers['feat_pcr_oi'].std(),
                'normal_mean': normal_trades['feat_pcr_oi'].mean() if len(normal_trades) > 0 else 0,
                'normal_std': normal_trades['feat_pcr_oi'].std() if len(normal_trades) > 0 else 0
            }
        
        # 8. Trade Type Analysis
        if 'type' in outliers.columns:
            type_dist_outliers = outliers['type'].value_counts(normalize=True)
            type_dist_normal = normal_trades['type'].value_counts(normalize=True) if len(normal_trades) > 0 else pd.Series()
            
            feature_analysis['trade_type'] = {
                'outlier_distribution': type_dist_outliers.to_dict(),
                'normal_distribution': type_dist_normal.to_dict() if len(type_dist_normal) > 0 else {}
            }
        
        analysis['features'] = feature_analysis
        self.analysis_results = analysis
        
        return analysis
    
    def get_outlier_summary(self) -> str:
        """Generate text summary of outlier analysis."""
        if not self.analysis_results:
            return "No analysis results available."
        
        a = self.analysis_results
        
        lines = [
            "=" * 60,
            "OUTLIER TRADE ANALYSIS SUMMARY",
            "=" * 60,
            "",
            f"Total Trades: {a['count']['total']}",
            f"Outlier Trades (Z > {self.z_threshold}): {a['count']['outliers']}",
            f"Normal Trades: {a['count']['normal']}",
            "",
            "P&L Statistics:",
            f"  Outlier Mean P&L: {a['pnl_stats']['outlier_mean']:.2f} points",
            f"  Outlier Range: {a['pnl_stats']['outlier_min']:.2f} to {a['pnl_stats']['outlier_max']:.2f}",
            f"  Normal Mean P&L: {a['pnl_stats']['normal_mean']:.2f} points",
            f"  Overall Mean: {a['pnl_stats']['all_mean']:.2f} points",
            f"  Overall Std: {a['pnl_stats']['all_std']:.2f} points",
            ""
        ]
        
        if 'features' in a:
            f = a['features']
            
            if 'regime' in f:
                lines.append("Regime Distribution (Outliers):")
                for regime, pct in f['regime']['outlier_distribution'].items():
                    regime_name = {1: 'Uptrend', 0: 'Sideways', -1: 'Downtrend'}.get(regime, str(regime))
                    lines.append(f"  {regime_name}: {pct*100:.1f}%")
                lines.append("")
            
            if 'time_of_day' in f:
                lines.append(f"Peak Hour for Outliers: {f['time_of_day'].get('outlier_peak_hour', 'N/A')}")
                lines.append("")
            
            if 'iv' in f:
                lines.append("IV Analysis:")
                lines.append(f"  Outlier Avg IV: {f['iv']['outlier_mean']:.2f}%")
                lines.append(f"  Normal Avg IV: {f['iv']['normal_mean']:.2f}%")
                lines.append("")
            
            if 'duration' in f:
                lines.append("Duration Analysis:")
                lines.append(f"  Outlier Mean Duration: {f['duration']['outlier_mean']:.1f} bars")
                lines.append(f"  Normal Mean Duration: {f['duration']['normal_mean']:.1f} bars")
                lines.append("")
            
            if 'ema_gap' in f:
                lines.append("EMA Gap Analysis:")
                lines.append(f"  Outlier Abs EMA Gap: {f['ema_gap']['outlier_abs_mean']:.2f}")
                lines.append(f"  Normal Abs EMA Gap: {f['ema_gap']['normal_abs_mean']:.2f}")
                lines.append("")
            
            if 'pcr' in f:
                lines.append("PCR Analysis:")
                lines.append(f"  Outlier PCR: {f['pcr']['outlier_mean']:.4f}")
                lines.append(f"  Normal PCR: {f['pcr']['normal_mean']:.4f}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_outlier_analysis(trade_log_path: Path, features_path: Path,
                          z_threshold: float = 3.0) -> Dict:
    """
    Run complete outlier analysis.
    
    Args:
        trade_log_path: Path to trade log CSV
        features_path: Path to features CSV
        z_threshold: Z-score threshold for outliers
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("OUTLIER DETECTION - Task 6.1")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading trade log from: {trade_log_path}")
    trade_log = pd.read_csv(trade_log_path)
    print(f"  Total trades: {len(trade_log)}")
    
    print(f"Loading features from: {features_path}")
    features_df = pd.read_csv(features_path)
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    print(f"  Total records: {len(features_df)}")
    
    # Initialize detector
    detector = OutlierDetector(z_threshold=z_threshold)
    
    # Calculate Z-scores and identify outliers
    print(f"\nIdentifying outliers (Z-score > {z_threshold})...")
    outliers = detector.identify_outliers(trade_log)
    
    print(f"  Positive outliers found: {len(outliers)}")
    print(f"  Negative outliers found: {len(detector.outliers['negative'])}")
    
    # Merge with features
    print("\nMerging trades with features...")
    all_trades_with_features = detector.merge_trade_features(
        detector.outliers['all_trades'], features_df
    )
    
    outliers_with_features = all_trades_with_features[
        all_trades_with_features['z_score'] > z_threshold
    ]
    
    print(f"  Trades with features: {len(all_trades_with_features)}")
    print(f"  Outliers with features: {len(outliers_with_features)}")
    
    # Analyze features
    print("\nAnalyzing outlier features...")
    analysis = detector.analyze_outlier_features(
        outliers_with_features, all_trades_with_features
    )
    
    # Print summary
    print("\n" + detector.get_outlier_summary())
    
    # Print outlier trades
    if len(outliers) > 0:
        print("\nOUTLIER TRADES (Z-score > 3):")
        print("-" * 80)
        outlier_display = outliers[['entry_time', 'exit_time', 'type', 'pnl', 'pnl_pct', 'z_score']].copy()
        outlier_display['pnl'] = outlier_display['pnl'].round(2)
        outlier_display['pnl_pct'] = outlier_display['pnl_pct'].round(4)
        outlier_display['z_score'] = outlier_display['z_score'].round(2)
        print(outlier_display.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS COMPLETE")
    print("=" * 70)
    
    return {
        'outliers': outliers,
        'outliers_with_features': outliers_with_features,
        'all_trades_with_features': all_trades_with_features,
        'analysis': analysis,
        'detector': detector
    }


def main():
    """Main function."""
    trade_log_path = Path("data/processed/trade_log.csv")
    features_path = Path("data/processed/nifty_features_5min.csv")
    
    results = run_outlier_analysis(trade_log_path, features_path, z_threshold=3.0)
    
    # Save results
    output_path = Path("data/processed/outlier_analysis.pkl")
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
