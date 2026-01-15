"""
Pattern Recognition Module - Task 6.2

Compare outlier trades vs normal profitable trades using statistical tests.
Create visualizations:
- Scatter plot (PnL vs duration)
- Box plots (feature distributions)
- Correlation heatmap
- Time distribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PatternRecognizer:
    """
    Compare outlier trades vs normal profitable trades.
    Uses statistical tests and visualizations to identify patterns.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize pattern recognizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or Path("data/processed/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.statistical_tests = {}
    
    def separate_trade_groups(self, trades_df: pd.DataFrame, 
                               z_threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Separate trades into outliers, normal profitable, and losing trades.
        
        Args:
            trades_df: DataFrame with all trades and z_score column
            z_threshold: Z-score threshold for outliers
            
        Returns:
            Tuple of (outliers, normal_profitable, losing)
        """
        outliers = trades_df[trades_df['z_score'] > z_threshold].copy()
        normal_profitable = trades_df[
            (trades_df['z_score'] <= z_threshold) & (trades_df['pnl'] > 0)
        ].copy()
        losing = trades_df[trades_df['pnl'] <= 0].copy()
        
        return outliers, normal_profitable, losing
    
    def run_statistical_tests(self, outliers: pd.DataFrame, 
                               normal_profitable: pd.DataFrame) -> Dict:
        """
        Run statistical tests comparing outliers vs normal profitable trades.
        
        Tests:
        - t-test: Compare means
        - Mann-Whitney U: Non-parametric comparison
        - Kolmogorov-Smirnov: Distribution comparison
        
        Args:
            outliers: DataFrame of outlier trades
            normal_profitable: DataFrame of normal profitable trades
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Features to test
        features_to_test = [
            ('pnl', 'P&L'),
            ('duration_bars', 'Duration'),
            ('feat_ema_diff', 'EMA Gap'),
            ('feat_avg_iv', 'Avg IV'),
            ('feat_pcr_oi', 'PCR (OI)'),
            ('feat_greeks_ce_atm_delta', 'Delta'),
            ('feat_greeks_ce_atm_gamma', 'Gamma'),
            ('feat_greeks_ce_atm_vega', 'Vega'),
        ]
        
        print("\nStatistical Tests: Outliers vs Normal Profitable")
        print("=" * 70)
        
        for col, name in features_to_test:
            if col not in outliers.columns or col not in normal_profitable.columns:
                continue
            
            outlier_vals = outliers[col].dropna()
            normal_vals = normal_profitable[col].dropna()
            
            if len(outlier_vals) < 2 or len(normal_vals) < 2:
                continue
            
            # t-test
            t_stat, t_pval = stats.ttest_ind(outlier_vals, normal_vals)
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_pval = stats.mannwhitneyu(outlier_vals, normal_vals, alternative='two-sided')
            except:
                u_stat, u_pval = np.nan, np.nan
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pval = stats.ks_2samp(outlier_vals, normal_vals)
            except:
                ks_stat, ks_pval = np.nan, np.nan
            
            results[col] = {
                'name': name,
                'outlier_mean': outlier_vals.mean(),
                'outlier_std': outlier_vals.std(),
                'normal_mean': normal_vals.mean(),
                'normal_std': normal_vals.std(),
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'mann_whitney_u': u_stat,
                'mann_whitney_pvalue': u_pval,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'significant': t_pval < 0.05 if not np.isnan(t_pval) else False
            }
            
            sig = "***" if t_pval < 0.01 else "**" if t_pval < 0.05 else "*" if t_pval < 0.1 else ""
            print(f"\n{name}:")
            print(f"  Outlier: {outlier_vals.mean():.4f} ± {outlier_vals.std():.4f}")
            print(f"  Normal:  {normal_vals.mean():.4f} ± {normal_vals.std():.4f}")
            print(f"  t-test p-value: {t_pval:.4f} {sig}")
            print(f"  Mann-Whitney p-value: {u_pval:.4f}")
        
        self.statistical_tests = results
        return results
    
    def plot_pnl_vs_duration(self, trades_df: pd.DataFrame, 
                              z_threshold: float = 3.0,
                              save: bool = True) -> plt.Figure:
        """
        Create scatter plot of PnL vs Duration.
        
        Args:
            trades_df: DataFrame with all trades
            z_threshold: Z-score threshold for outliers
            save: Whether to save the plot
            
        Returns:
            matplotlib Figure
        """
        outliers, normal_profitable, losing = self.separate_trade_groups(trades_df, z_threshold)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot losing trades
        if len(losing) > 0:
            ax.scatter(losing['duration_bars'], losing['pnl'], 
                      c='#e74c3c', alpha=0.5, s=50, label=f'Losing ({len(losing)})')
        
        # Plot normal profitable
        if len(normal_profitable) > 0:
            ax.scatter(normal_profitable['duration_bars'], normal_profitable['pnl'],
                      c='#3498db', alpha=0.6, s=60, label=f'Normal Profitable ({len(normal_profitable)})')
        
        # Plot outliers
        if len(outliers) > 0:
            ax.scatter(outliers['duration_bars'], outliers['pnl'],
                      c='#2ecc71', alpha=0.9, s=150, marker='*', 
                      edgecolors='black', linewidths=1,
                      label=f'Outliers Z>{z_threshold} ({len(outliers)})')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add 3-sigma line
        mean_pnl = trades_df['pnl'].mean()
        std_pnl = trades_df['pnl'].std()
        ax.axhline(y=mean_pnl + z_threshold * std_pnl, color='green', 
                  linestyle=':', alpha=0.7, label=f'{z_threshold}σ threshold')
        
        ax.set_xlabel('Duration (bars)', fontsize=12)
        ax.set_ylabel('P&L (points)', fontsize=12)
        ax.set_title('P&L vs Trade Duration', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "pnl_vs_duration_scatter.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_feature_boxplots(self, trades_df: pd.DataFrame,
                               z_threshold: float = 3.0,
                               save: bool = True) -> plt.Figure:
        """
        Create box plots comparing feature distributions.
        
        Args:
            trades_df: DataFrame with all trades
            z_threshold: Z-score threshold for outliers
            save: Whether to save the plot
            
        Returns:
            matplotlib Figure
        """
        outliers, normal_profitable, losing = self.separate_trade_groups(trades_df, z_threshold)
        
        # Features to plot
        features = [
            ('pnl', 'P&L'),
            ('duration_bars', 'Duration'),
            ('feat_ema_diff', 'EMA Gap'),
            ('feat_avg_iv', 'Avg IV'),
            ('feat_pcr_oi', 'PCR'),
            ('feat_greeks_ce_atm_delta', 'Delta'),
        ]
        
        # Filter to available features
        features = [(col, name) for col, name in features if col in trades_df.columns]
        
        n_features = len(features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (col, name) in enumerate(features):
            ax = axes[idx]
            
            # Prepare data for box plot
            data_to_plot = []
            labels = []
            colors = []
            
            if len(outliers) > 0 and col in outliers.columns:
                data_to_plot.append(outliers[col].dropna())
                labels.append('Outliers')
                colors.append('#2ecc71')
            
            if len(normal_profitable) > 0 and col in normal_profitable.columns:
                data_to_plot.append(normal_profitable[col].dropna())
                labels.append('Normal\nProfitable')
                colors.append('#3498db')
            
            if len(losing) > 0 and col in losing.columns:
                data_to_plot.append(losing[col].dropna())
                labels.append('Losing')
                colors.append('#e74c3c')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
        
        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions by Trade Category', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "feature_boxplots.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_correlation_heatmap(self, trades_df: pd.DataFrame,
                                  save: bool = True) -> plt.Figure:
        """
        Create correlation heatmap of trade features.
        
        Args:
            trades_df: DataFrame with all trades
            save: Whether to save the plot
            
        Returns:
            matplotlib Figure
        """
        # Select numeric columns for correlation
        corr_cols = [
            'pnl', 'pnl_pct', 'duration_bars', 'z_score',
            'feat_ema_diff', 'feat_avg_iv', 'feat_pcr_oi',
            'feat_greeks_ce_atm_delta', 'feat_greeks_ce_atm_gamma',
            'feat_greeks_ce_atm_vega', 'feat_hour'
        ]
        
        # Filter to available columns
        corr_cols = [col for col in corr_cols if col in trades_df.columns]
        
        # Calculate correlation matrix
        corr_matrix = trades_df[corr_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Rename columns for display
        display_names = {
            'pnl': 'P&L',
            'pnl_pct': 'P&L %',
            'duration_bars': 'Duration',
            'z_score': 'Z-Score',
            'feat_ema_diff': 'EMA Gap',
            'feat_avg_iv': 'Avg IV',
            'feat_pcr_oi': 'PCR',
            'feat_greeks_ce_atm_delta': 'Delta',
            'feat_greeks_ce_atm_gamma': 'Gamma',
            'feat_greeks_ce_atm_vega': 'Vega',
            'feat_hour': 'Hour'
        }
        
        corr_matrix_display = corr_matrix.rename(columns=display_names, index=display_names)
        
        sns.heatmap(corr_matrix_display, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 9})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "correlation_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_time_distribution(self, trades_df: pd.DataFrame,
                                z_threshold: float = 3.0,
                                save: bool = True) -> plt.Figure:
        """
        Create time distribution visualization.
        
        Args:
            trades_df: DataFrame with all trades
            z_threshold: Z-score threshold for outliers
            save: Whether to save the plot
            
        Returns:
            matplotlib Figure
        """
        outliers, normal_profitable, losing = self.separate_trade_groups(trades_df, z_threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Hour distribution
        ax1 = axes[0, 0]
        
        if 'feat_hour' in trades_df.columns:
            hours = range(9, 16)
            
            outlier_hours = outliers['feat_hour'].value_counts().reindex(hours, fill_value=0)
            normal_hours = normal_profitable['feat_hour'].value_counts().reindex(hours, fill_value=0)
            losing_hours = losing['feat_hour'].value_counts().reindex(hours, fill_value=0)
            
            x = np.arange(len(hours))
            width = 0.25
            
            ax1.bar(x - width, outlier_hours.values, width, label='Outliers', color='#2ecc71', alpha=0.7)
            ax1.bar(x, normal_hours.values, width, label='Normal Profitable', color='#3498db', alpha=0.7)
            ax1.bar(x + width, losing_hours.values, width, label='Losing', color='#e74c3c', alpha=0.7)
            
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Count')
            ax1.set_title('Trade Distribution by Hour', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(hours)
            ax1.legend()
        
        # 2. Day of week distribution
        ax2 = axes[0, 1]
        
        if 'feat_day_of_week' in trades_df.columns:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            
            outlier_days = outliers['feat_day_of_week'].value_counts().reindex(range(5), fill_value=0)
            normal_days = normal_profitable['feat_day_of_week'].value_counts().reindex(range(5), fill_value=0)
            losing_days = losing['feat_day_of_week'].value_counts().reindex(range(5), fill_value=0)
            
            x = np.arange(len(days))
            width = 0.25
            
            ax2.bar(x - width, outlier_days.values, width, label='Outliers', color='#2ecc71', alpha=0.7)
            ax2.bar(x, normal_days.values, width, label='Normal Profitable', color='#3498db', alpha=0.7)
            ax2.bar(x + width, losing_days.values, width, label='Losing', color='#e74c3c', alpha=0.7)
            
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Count')
            ax2.set_title('Trade Distribution by Day', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(days)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Day of week data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. P&L distribution by hour (heatmap style)
        ax3 = axes[1, 0]
        
        if 'feat_hour' in trades_df.columns:
            # Calculate mean P&L by hour for each group
            pnl_by_hour = trades_df.groupby('feat_hour')['pnl'].agg(['mean', 'count'])
            
            ax3.bar(pnl_by_hour.index, pnl_by_hour['mean'], 
                   color=['#2ecc71' if x > 0 else '#e74c3c' for x in pnl_by_hour['mean']],
                   alpha=0.7)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Hour')
            ax3.set_ylabel('Mean P&L')
            ax3.set_title('Average P&L by Hour', fontweight='bold')
        
        # 4. Cumulative P&L over time
        ax4 = axes[1, 1]
        
        if 'entry_time' in trades_df.columns:
            trades_sorted = trades_df.sort_values('entry_time')
            trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
            
            ax4.plot(range(len(trades_sorted)), trades_sorted['cumulative_pnl'], 
                    color='#2c3e50', linewidth=2)
            
            # Mark outlier trades
            outlier_indices = trades_sorted[trades_sorted['z_score'] > z_threshold].index
            for idx in outlier_indices:
                pos = trades_sorted.index.get_loc(idx)
                ax4.scatter(pos, trades_sorted.loc[idx, 'cumulative_pnl'],
                           c='#2ecc71', s=100, marker='*', zorder=5)
            
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative P&L')
            ax4.set_title('Cumulative P&L with Outliers Marked', fontweight='bold')
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle('Time-Based Trade Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "time_distribution.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def create_all_visualizations(self, trades_df: pd.DataFrame,
                                   z_threshold: float = 3.0) -> Dict[str, Path]:
        """
        Create all visualizations.
        
        Args:
            trades_df: DataFrame with all trades
            z_threshold: Z-score threshold for outliers
            
        Returns:
            Dictionary of plot names to file paths
        """
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        paths = {}
        
        # 1. PnL vs Duration scatter
        print("\n1. Creating P&L vs Duration scatter plot...")
        self.plot_pnl_vs_duration(trades_df, z_threshold)
        paths['scatter'] = self.output_dir / "pnl_vs_duration_scatter.png"
        
        # 2. Feature box plots
        print("\n2. Creating feature box plots...")
        self.plot_feature_boxplots(trades_df, z_threshold)
        paths['boxplots'] = self.output_dir / "feature_boxplots.png"
        
        # 3. Correlation heatmap
        print("\n3. Creating correlation heatmap...")
        self.plot_correlation_heatmap(trades_df)
        paths['heatmap'] = self.output_dir / "correlation_heatmap.png"
        
        # 4. Time distribution
        print("\n4. Creating time distribution plots...")
        self.plot_time_distribution(trades_df, z_threshold)
        paths['time'] = self.output_dir / "time_distribution.png"
        
        return paths


def run_pattern_recognition(outlier_analysis_path: Path) -> Dict:
    """
    Run complete pattern recognition analysis.
    
    Args:
        outlier_analysis_path: Path to outlier analysis pickle file
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("PATTERN RECOGNITION - Task 6.2")
    print("=" * 70)
    
    # Load outlier analysis results
    print(f"\nLoading outlier analysis from: {outlier_analysis_path}")
    with open(outlier_analysis_path, 'rb') as f:
        outlier_data = pickle.load(f)
    
    trades_df = outlier_data['all_trades_with_features']
    print(f"  Total trades with features: {len(trades_df)}")
    
    # Initialize pattern recognizer
    recognizer = PatternRecognizer()
    
    # Separate trade groups
    outliers, normal_profitable, losing = recognizer.separate_trade_groups(trades_df, z_threshold=3.0)
    print(f"\nTrade Groups:")
    print(f"  Outliers (Z > 3): {len(outliers)}")
    print(f"  Normal Profitable: {len(normal_profitable)}")
    print(f"  Losing: {len(losing)}")
    
    # Run statistical tests
    stat_results = recognizer.run_statistical_tests(outliers, normal_profitable)
    
    # Create visualizations
    viz_paths = recognizer.create_all_visualizations(trades_df, z_threshold=3.0)
    
    print("\n" + "=" * 70)
    print("PATTERN RECOGNITION COMPLETE")
    print("=" * 70)
    
    print(f"\nVisualization files saved to: {recognizer.output_dir}")
    for name, path in viz_paths.items():
        print(f"  - {name}: {path.name}")
    
    return {
        'statistical_tests': stat_results,
        'visualization_paths': viz_paths,
        'trade_groups': {
            'outliers': len(outliers),
            'normal_profitable': len(normal_profitable),
            'losing': len(losing)
        }
    }


def main():
    """Main function."""
    outlier_analysis_path = Path("data/processed/outlier_analysis.pkl")
    
    results = run_pattern_recognition(outlier_analysis_path)
    
    # Save results
    output_path = Path("data/processed/pattern_recognition.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
