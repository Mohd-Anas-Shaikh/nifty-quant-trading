"""
Regime Visualization Module - Task 3.2

This module creates visualizations for regime analysis:
1. Price chart with regime overlay (color-coded)
2. Transition matrix heatmap
3. Regime statistics (IV, Greeks distribution per regime)
4. Duration histogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class RegimeVisualizer:
    """
    Creates visualizations for regime analysis.
    """
    
    # Regime colors
    REGIME_COLORS = {
        1: '#2ecc71',    # Green for Uptrend
        0: '#f39c12',    # Orange for Sideways
        -1: '#e74c3c'    # Red for Downtrend
    }
    
    REGIME_NAMES = {
        1: 'Uptrend',
        0: 'Sideways',
        -1: 'Downtrend'
    }
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir or Path("data/processed/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_price_with_regimes(self, df: pd.DataFrame, 
                                 save: bool = True,
                                 figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        Create price chart with regime overlay (color-coded background).
        
        Args:
            df: DataFrame with price and regime data
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Filter to rows with regime data
        df_regime = df.dropna(subset=['regime']).copy()
        
        if len(df_regime) == 0:
            print("No regime data available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price line
        ax.plot(df_regime['timestamp'], df_regime['spot_close'], 
                color='#2c3e50', linewidth=1, label='NIFTY 50 Spot', zorder=3)
        
        # Add regime background colors
        regime_changes = df_regime['regime'].ne(df_regime['regime'].shift()).cumsum()
        
        for _, group in df_regime.groupby(regime_changes):
            regime = group['regime'].iloc[0]
            color = self.REGIME_COLORS.get(regime, '#95a5a6')
            
            ax.axvspan(group['timestamp'].iloc[0], 
                      group['timestamp'].iloc[-1],
                      alpha=0.3, color=color, zorder=1)
        
        # Add legend for regimes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.REGIME_COLORS[1], alpha=0.3, label='Uptrend (+1)'),
            Patch(facecolor=self.REGIME_COLORS[0], alpha=0.3, label='Sideways (0)'),
            Patch(facecolor=self.REGIME_COLORS[-1], alpha=0.3, label='Downtrend (-1)'),
            plt.Line2D([0], [0], color='#2c3e50', linewidth=2, label='NIFTY 50')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('NIFTY 50 Price with HMM Regime Overlay', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "regime_price_overlay.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_transition_matrix(self, model_path: Path = None,
                                save: bool = True,
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create transition matrix heatmap.
        
        Args:
            model_path: Path to saved HMM model
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Load model to get transition matrix
        model_path = model_path or Path("data/processed/hmm_regime_model.pkl")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trans_mat = model_data['model'].transmat_
        regime_map = model_data['regime_map']
        
        # Reorder matrix to match regime order (-1, 0, +1)
        state_to_regime = {v: k for k, v in regime_map.items()}
        ordered_states = [state_to_regime[-1], state_to_regime[0], state_to_regime[1]]
        
        # Create reordered matrix
        n_states = len(ordered_states)
        reordered_mat = np.zeros((n_states, n_states))
        
        for i, from_state in enumerate(ordered_states):
            for j, to_state in enumerate(ordered_states):
                reordered_mat[i, j] = trans_mat[from_state, to_state]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = ['Downtrend (-1)', 'Sideways (0)', 'Uptrend (+1)']
        
        sns.heatmap(reordered_mat, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, vmin=0, vmax=1,
                    annot_kws={'size': 12})
        
        ax.set_xlabel('To Regime', fontsize=12)
        ax.set_ylabel('From Regime', fontsize=12)
        ax.set_title('Regime Transition Probability Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "transition_matrix_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_regime_statistics(self, df: pd.DataFrame,
                                save: bool = True,
                                figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create regime statistics plots (IV, Greeks distribution per regime).
        
        Args:
            df: DataFrame with regime and feature data
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Filter to rows with regime data
        df_regime = df.dropna(subset=['regime']).copy()
        
        if len(df_regime) == 0:
            print("No regime data available for plotting")
            return None
        
        # Features to plot
        features = [
            ('avg_iv', 'Average IV (%)'),
            ('iv_spread', 'IV Spread (%)'),
            ('pcr_oi', 'Put-Call Ratio (OI)'),
            ('greeks_ce_atm_delta', 'ATM Call Delta'),
            ('greeks_ce_atm_gamma', 'ATM Call Gamma'),
            ('greeks_ce_atm_vega', 'ATM Call Vega')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (feat, title) in enumerate(features):
            ax = axes[idx]
            
            if feat not in df_regime.columns:
                ax.text(0.5, 0.5, f'{feat} not available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create box plot for each regime
            data_by_regime = []
            labels = []
            colors = []
            
            for regime in [-1, 0, 1]:
                regime_data = df_regime[df_regime['regime'] == regime][feat].dropna()
                if len(regime_data) > 0:
                    data_by_regime.append(regime_data)
                    labels.append(self.REGIME_NAMES[regime])
                    colors.append(self.REGIME_COLORS[regime])
            
            if data_by_regime:
                bp = ax.boxplot(data_by_regime, labels=labels, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=15)
        
        plt.suptitle('Feature Distribution by Regime', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "regime_statistics.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_duration_histogram(self, df: pd.DataFrame,
                                 save: bool = True,
                                 figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """
        Create regime duration histogram.
        
        Args:
            df: DataFrame with regime data
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Filter to rows with regime data
        df_regime = df.dropna(subset=['regime']).copy()
        
        if len(df_regime) == 0:
            print("No regime data available for plotting")
            return None
        
        # Calculate regime durations
        durations = {-1: [], 0: [], 1: []}
        
        # Identify regime segments
        df_regime['regime_change'] = df_regime['regime'].ne(df_regime['regime'].shift()).cumsum()
        
        for _, group in df_regime.groupby('regime_change'):
            regime = group['regime'].iloc[0]
            duration = len(group)  # Duration in number of observations (hours since hourly data)
            durations[regime].append(duration)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, regime in enumerate([-1, 0, 1]):
            ax = axes[idx]
            
            if durations[regime]:
                ax.hist(durations[regime], bins=20, 
                       color=self.REGIME_COLORS[regime], 
                       alpha=0.7, edgecolor='black')
                
                # Add statistics
                mean_dur = np.mean(durations[regime])
                median_dur = np.median(durations[regime])
                max_dur = np.max(durations[regime])
                
                stats_text = f'Mean: {mean_dur:.1f}\nMedian: {median_dur:.1f}\nMax: {max_dur}'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)
            
            ax.set_title(f'{self.REGIME_NAMES[regime]} Duration', fontsize=12, fontweight='bold')
            ax.set_xlabel('Duration (observations)')
            ax.set_ylabel('Frequency')
        
        plt.suptitle('Regime Duration Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "regime_duration_histogram.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_regime_pie_chart(self, df: pd.DataFrame,
                               save: bool = True,
                               figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Create pie chart of regime distribution.
        """
        df_regime = df.dropna(subset=['regime']).copy()
        
        if len(df_regime) == 0:
            return None
        
        regime_counts = df_regime['regime'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = [self.REGIME_COLORS[r] for r in regime_counts.index]
        labels = [f"{self.REGIME_NAMES[r]}\n({regime_counts[r]} obs)" for r in regime_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            regime_counts.values, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.02] * len(regime_counts)
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Regime Distribution', fontsize=14, fontweight='bold')
        
        if save:
            path = self.output_dir / "regime_pie_chart.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def create_all_visualizations(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Create all regime visualizations.
        
        Args:
            df: DataFrame with regime data
            
        Returns:
            Dictionary of plot names to file paths
        """
        print("=" * 60)
        print("REGIME VISUALIZATION - Task 3.2")
        print("=" * 60)
        
        paths = {}
        
        # 1. Price with regime overlay
        print("\n1. Creating price chart with regime overlay...")
        self.plot_price_with_regimes(df)
        paths['price_overlay'] = self.output_dir / "regime_price_overlay.png"
        
        # 2. Transition matrix heatmap
        print("\n2. Creating transition matrix heatmap...")
        self.plot_transition_matrix()
        paths['transition_matrix'] = self.output_dir / "transition_matrix_heatmap.png"
        
        # 3. Regime statistics
        print("\n3. Creating regime statistics plots...")
        self.plot_regime_statistics(df)
        paths['statistics'] = self.output_dir / "regime_statistics.png"
        
        # 4. Duration histogram
        print("\n4. Creating duration histogram...")
        self.plot_duration_histogram(df)
        paths['duration'] = self.output_dir / "regime_duration_histogram.png"
        
        # 5. Pie chart (bonus)
        print("\n5. Creating regime pie chart...")
        self.plot_regime_pie_chart(df)
        paths['pie_chart'] = self.output_dir / "regime_pie_chart.png"
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"\nAll plots saved to: {self.output_dir}")
        
        return paths


def create_regime_visualizations(input_path: Path) -> Dict[str, Path]:
    """
    Main function to create all regime visualizations.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        Dictionary of plot names to file paths
    """
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    print(f"  Records with regime: {df['regime'].notna().sum()}")
    
    # Create visualizer
    visualizer = RegimeVisualizer()
    
    # Create all visualizations
    paths = visualizer.create_all_visualizations(df)
    
    return paths


def main():
    """Main function."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    paths = create_regime_visualizations(input_path)
    
    print("\nGenerated visualizations:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
