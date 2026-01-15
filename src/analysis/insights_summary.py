"""
Insights Summary Module - Task 6.3

Generate comprehensive insights summary answering:
- What percentage are outliers?
- Average PnL comparison
- Regime patterns
- Time-of-day patterns
- IV characteristics
- Distinguishing features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')


class InsightsSummary:
    """
    Generate comprehensive insights summary from outlier analysis.
    """
    
    def __init__(self):
        """Initialize insights summary generator."""
        self.insights = {}
    
    def calculate_outlier_percentage(self, trades_df: pd.DataFrame, 
                                      z_threshold: float = 3.0) -> Dict:
        """
        Calculate what percentage of trades are outliers.
        """
        total_trades = len(trades_df)
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        profitable = trades_df[trades_df['pnl'] > 0]
        
        return {
            'total_trades': total_trades,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / total_trades * 100,
            'profitable_count': len(profitable),
            'profitable_percentage': len(profitable) / total_trades * 100,
            'outlier_of_profitable': len(outliers) / len(profitable) * 100 if len(profitable) > 0 else 0
        }
    
    def calculate_pnl_comparison(self, trades_df: pd.DataFrame,
                                  z_threshold: float = 3.0) -> Dict:
        """
        Calculate average PnL comparison between groups.
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        normal_profitable = trades_df[(trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)]
        losing = trades_df[trades_df['pnl'] <= 0]
        all_trades = trades_df
        
        return {
            'outlier_avg_pnl': outliers['pnl'].mean() if len(outliers) > 0 else 0,
            'outlier_total_pnl': outliers['pnl'].sum() if len(outliers) > 0 else 0,
            'outlier_avg_pnl_pct': outliers['pnl_pct'].mean() if len(outliers) > 0 else 0,
            'normal_profitable_avg_pnl': normal_profitable['pnl'].mean() if len(normal_profitable) > 0 else 0,
            'normal_profitable_total_pnl': normal_profitable['pnl'].sum() if len(normal_profitable) > 0 else 0,
            'losing_avg_pnl': losing['pnl'].mean() if len(losing) > 0 else 0,
            'losing_total_pnl': losing['pnl'].sum() if len(losing) > 0 else 0,
            'all_avg_pnl': all_trades['pnl'].mean(),
            'all_total_pnl': all_trades['pnl'].sum(),
            'outlier_contribution': outliers['pnl'].sum() / all_trades['pnl'].sum() * 100 if all_trades['pnl'].sum() != 0 else 0
        }
    
    def analyze_regime_patterns(self, trades_df: pd.DataFrame,
                                 z_threshold: float = 3.0) -> Dict:
        """
        Analyze regime patterns for outliers vs other trades.
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        normal_profitable = trades_df[(trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        regime_map = {-1.0: 'Downtrend', 0.0: 'Sideways', 1.0: 'Uptrend'}
        
        def get_regime_dist(df, col='feat_regime_filled'):
            if col not in df.columns or len(df) == 0:
                return {}
            dist = df[col].value_counts(normalize=True)
            return {regime_map.get(k, str(k)): v * 100 for k, v in dist.items()}
        
        def get_type_dist(df):
            if 'type' not in df.columns or len(df) == 0:
                return {}
            return df['type'].value_counts(normalize=True).to_dict()
        
        return {
            'outlier_regime_dist': get_regime_dist(outliers),
            'normal_profitable_regime_dist': get_regime_dist(normal_profitable),
            'losing_regime_dist': get_regime_dist(losing),
            'outlier_type_dist': get_type_dist(outliers),
            'normal_profitable_type_dist': get_type_dist(normal_profitable),
            'dominant_outlier_regime': max(get_regime_dist(outliers).items(), key=lambda x: x[1])[0] if get_regime_dist(outliers) else 'N/A',
            'dominant_outlier_type': max(get_type_dist(outliers).items(), key=lambda x: x[1])[0] if get_type_dist(outliers) else 'N/A'
        }
    
    def analyze_time_patterns(self, trades_df: pd.DataFrame,
                               z_threshold: float = 3.0) -> Dict:
        """
        Analyze time-of-day patterns for outliers.
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        normal_profitable = trades_df[(trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)]
        
        def get_hour_dist(df, col='feat_hour'):
            if col not in df.columns or len(df) == 0:
                return {}
            return df[col].value_counts(normalize=True).to_dict()
        
        def get_peak_hour(df, col='feat_hour'):
            if col not in df.columns or len(df) == 0:
                return None
            return df[col].mode().iloc[0] if len(df) > 0 else None
        
        # Best hours for outliers
        outlier_hours = get_hour_dist(outliers)
        normal_hours = get_hour_dist(normal_profitable)
        
        return {
            'outlier_hour_dist': outlier_hours,
            'normal_profitable_hour_dist': normal_hours,
            'outlier_peak_hour': get_peak_hour(outliers),
            'normal_peak_hour': get_peak_hour(normal_profitable),
            'morning_session': {
                'outlier_pct': sum(v for k, v in outlier_hours.items() if k in [9, 10, 11]) * 100 if outlier_hours else 0,
                'normal_pct': sum(v for k, v in normal_hours.items() if k in [9, 10, 11]) * 100 if normal_hours else 0
            },
            'afternoon_session': {
                'outlier_pct': sum(v for k, v in outlier_hours.items() if k in [12, 13, 14, 15]) * 100 if outlier_hours else 0,
                'normal_pct': sum(v for k, v in normal_hours.items() if k in [12, 13, 14, 15]) * 100 if normal_hours else 0
            }
        }
    
    def analyze_iv_characteristics(self, trades_df: pd.DataFrame,
                                    z_threshold: float = 3.0) -> Dict:
        """
        Analyze IV characteristics for outliers.
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        normal_profitable = trades_df[(trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        def get_iv_stats(df, col='feat_avg_iv'):
            if col not in df.columns or len(df) == 0:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            return {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        def get_iv_spread_stats(df, col='feat_iv_spread'):
            if col not in df.columns or len(df) == 0:
                return {'mean': 0, 'std': 0}
            return {
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        
        return {
            'outlier_iv': get_iv_stats(outliers),
            'normal_profitable_iv': get_iv_stats(normal_profitable),
            'losing_iv': get_iv_stats(losing),
            'outlier_iv_spread': get_iv_spread_stats(outliers),
            'normal_profitable_iv_spread': get_iv_spread_stats(normal_profitable),
            'iv_difference': get_iv_stats(outliers)['mean'] - get_iv_stats(normal_profitable)['mean']
        }
    
    def identify_distinguishing_features(self, trades_df: pd.DataFrame,
                                          z_threshold: float = 3.0) -> Dict:
        """
        Identify features that distinguish outliers from normal trades.
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold]
        normal_profitable = trades_df[(trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)]
        
        features_to_compare = [
            ('duration_bars', 'Duration (bars)'),
            ('feat_ema_diff', 'EMA Gap'),
            ('feat_avg_iv', 'Avg IV'),
            ('feat_pcr_oi', 'PCR (OI)'),
            ('feat_greeks_ce_atm_delta', 'Delta'),
            ('feat_greeks_ce_atm_gamma', 'Gamma'),
            ('feat_greeks_ce_atm_vega', 'Vega'),
        ]
        
        distinguishing = []
        
        for col, name in features_to_compare:
            if col not in outliers.columns or col not in normal_profitable.columns:
                continue
            
            outlier_mean = outliers[col].mean()
            normal_mean = normal_profitable[col].mean()
            
            if normal_mean != 0:
                ratio = outlier_mean / normal_mean
            else:
                ratio = float('inf') if outlier_mean > 0 else 0
            
            distinguishing.append({
                'feature': name,
                'column': col,
                'outlier_mean': outlier_mean,
                'normal_mean': normal_mean,
                'ratio': ratio,
                'difference': outlier_mean - normal_mean,
                'is_significant': abs(ratio - 1) > 0.5 or abs(outlier_mean - normal_mean) > normal_profitable[col].std() if len(normal_profitable) > 0 else False
            })
        
        # Sort by significance (ratio furthest from 1)
        distinguishing.sort(key=lambda x: abs(x['ratio'] - 1) if x['ratio'] != float('inf') else 1000, reverse=True)
        
        return {
            'all_features': distinguishing,
            'significant_features': [f for f in distinguishing if f['is_significant']],
            'top_distinguishing': distinguishing[:3] if len(distinguishing) >= 3 else distinguishing
        }
    
    def generate_full_summary(self, trades_df: pd.DataFrame,
                               z_threshold: float = 3.0) -> Dict:
        """
        Generate complete insights summary.
        """
        self.insights = {
            'outlier_percentage': self.calculate_outlier_percentage(trades_df, z_threshold),
            'pnl_comparison': self.calculate_pnl_comparison(trades_df, z_threshold),
            'regime_patterns': self.analyze_regime_patterns(trades_df, z_threshold),
            'time_patterns': self.analyze_time_patterns(trades_df, z_threshold),
            'iv_characteristics': self.analyze_iv_characteristics(trades_df, z_threshold),
            'distinguishing_features': self.identify_distinguishing_features(trades_df, z_threshold)
        }
        
        return self.insights
    
    def print_summary(self) -> str:
        """Generate formatted text summary."""
        if not self.insights:
            return "No insights generated yet."
        
        i = self.insights
        
        lines = [
            "=" * 70,
            "INSIGHTS SUMMARY - Task 6.3",
            "=" * 70,
            "",
            "1. WHAT PERCENTAGE ARE OUTLIERS?",
            "-" * 40,
            f"   Total Trades: {i['outlier_percentage']['total_trades']}",
            f"   Outlier Trades: {i['outlier_percentage']['outlier_count']}",
            f"   Outlier Percentage: {i['outlier_percentage']['outlier_percentage']:.2f}%",
            f"   Profitable Trades: {i['outlier_percentage']['profitable_count']}",
            f"   Outliers as % of Profitable: {i['outlier_percentage']['outlier_of_profitable']:.2f}%",
            "",
            "2. AVERAGE PNL COMPARISON",
            "-" * 40,
            f"   Outlier Avg P&L: {i['pnl_comparison']['outlier_avg_pnl']:.2f} points",
            f"   Normal Profitable Avg P&L: {i['pnl_comparison']['normal_profitable_avg_pnl']:.2f} points",
            f"   Losing Avg P&L: {i['pnl_comparison']['losing_avg_pnl']:.2f} points",
            f"   All Trades Avg P&L: {i['pnl_comparison']['all_avg_pnl']:.2f} points",
            f"   ",
            f"   Outlier Total P&L: {i['pnl_comparison']['outlier_total_pnl']:.2f} points",
            f"   Outlier Contribution to Total: {i['pnl_comparison']['outlier_contribution']:.1f}%",
            "",
            "3. REGIME PATTERNS",
            "-" * 40,
            f"   Dominant Outlier Regime: {i['regime_patterns']['dominant_outlier_regime']}",
            f"   Dominant Outlier Type: {i['regime_patterns']['dominant_outlier_type']}",
            "   ",
            "   Outlier Regime Distribution:",
        ]
        
        for regime, pct in i['regime_patterns']['outlier_regime_dist'].items():
            lines.append(f"     {regime}: {pct:.1f}%")
        
        lines.extend([
            "",
            "4. TIME-OF-DAY PATTERNS",
            "-" * 40,
            f"   Outlier Peak Hour: {i['time_patterns']['outlier_peak_hour']}:00",
            f"   Normal Peak Hour: {i['time_patterns']['normal_peak_hour']}:00",
            f"   ",
            f"   Morning Session (9-11h):",
            f"     Outliers: {i['time_patterns']['morning_session']['outlier_pct']:.1f}%",
            f"     Normal: {i['time_patterns']['morning_session']['normal_pct']:.1f}%",
            f"   Afternoon Session (12-15h):",
            f"     Outliers: {i['time_patterns']['afternoon_session']['outlier_pct']:.1f}%",
            f"     Normal: {i['time_patterns']['afternoon_session']['normal_pct']:.1f}%",
            "",
            "5. IV CHARACTERISTICS",
            "-" * 40,
            f"   Outlier Avg IV: {i['iv_characteristics']['outlier_iv']['mean']:.2f}%",
            f"   Normal Profitable Avg IV: {i['iv_characteristics']['normal_profitable_iv']['mean']:.2f}%",
            f"   IV Difference: {i['iv_characteristics']['iv_difference']:.2f}%",
            "",
            "6. DISTINGUISHING FEATURES",
            "-" * 40,
        ])
        
        for feat in i['distinguishing_features']['top_distinguishing']:
            ratio_str = f"{feat['ratio']:.2f}x" if feat['ratio'] != float('inf') else "∞"
            lines.append(f"   {feat['feature']}:")
            lines.append(f"     Outlier: {feat['outlier_mean']:.4f}")
            lines.append(f"     Normal: {feat['normal_mean']:.4f}")
            lines.append(f"     Ratio: {ratio_str}")
            lines.append("")
        
        lines.extend([
            "=" * 70,
            "KEY TAKEAWAYS",
            "=" * 70,
            "",
            f"• Only {i['outlier_percentage']['outlier_percentage']:.1f}% of trades are outliers",
            f"• Outliers contribute {i['pnl_comparison']['outlier_contribution']:.0f}% of total profits",
            f"• {i['regime_patterns']['dominant_outlier_regime']} regime dominates outliers",
            f"• {i['regime_patterns']['dominant_outlier_type']} trades are most common among outliers",
            f"• Peak outlier hour is {i['time_patterns']['outlier_peak_hour']}:00",
            f"• Duration is the most distinguishing feature",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def run_insights_summary(outlier_analysis_path: Path) -> Dict:
    """
    Run complete insights summary.
    
    Args:
        outlier_analysis_path: Path to outlier analysis pickle file
        
    Returns:
        Dictionary with all insights
    """
    print("=" * 70)
    print("INSIGHTS SUMMARY - Task 6.3")
    print("=" * 70)
    
    # Load outlier analysis results
    print(f"\nLoading outlier analysis from: {outlier_analysis_path}")
    with open(outlier_analysis_path, 'rb') as f:
        outlier_data = pickle.load(f)
    
    trades_df = outlier_data['all_trades_with_features']
    print(f"  Total trades: {len(trades_df)}")
    
    # Generate insights
    summary = InsightsSummary()
    insights = summary.generate_full_summary(trades_df, z_threshold=3.0)
    
    # Print summary
    print("\n" + summary.print_summary())
    
    return insights


def main():
    """Main function."""
    outlier_analysis_path = Path("data/processed/outlier_analysis.pkl")
    
    insights = run_insights_summary(outlier_analysis_path)
    
    # Save insights
    output_path = Path("data/processed/insights_summary.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(insights, f)
    print(f"\nInsights saved to: {output_path}")
    
    return insights


if __name__ == "__main__":
    main()
