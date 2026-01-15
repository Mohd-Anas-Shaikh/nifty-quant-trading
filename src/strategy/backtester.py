"""
Backtesting Module - Task 4.2

This module provides comprehensive backtesting with:
- Train/Test split (70/30)
- Performance metrics: Total Return, Sharpe, Sortino, Calmar, Max Drawdown
- Trade metrics: Win Rate, Profit Factor, Avg Duration, Total Trades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import warnings

from .ema_regime_strategy import EMARegimeStrategy, Trade

warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results."""
    period: str  # 'train', 'test', or 'full'
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict


class PerformanceMetrics:
    """
    Calculate trading performance metrics.
    
    Metrics calculated:
    - Total Return
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Maximum Drawdown
    - Win Rate
    - Profit Factor
    - Average Trade Duration
    - Total Trades
    """
    
    def __init__(self, risk_free_rate: float = 0.065, periods_per_year: int = 252 * 75):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 6.5%)
            periods_per_year: Number of trading periods per year (5-min bars)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate period returns from equity curve."""
        return equity_curve.pct_change().dropna()
    
    def total_return(self, equity_curve: pd.Series) -> float:
        """
        Calculate total return.
        
        Total Return = (Final Value - Initial Value) / Initial Value × 100
        """
        if len(equity_curve) < 2:
            return 0.0
        return ((equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]) * 100
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        
        Annualized using periods_per_year.
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        # Annualize
        mean_return = returns.mean() * self.periods_per_year
        std_return = returns.std() * np.sqrt(self.periods_per_year)
        
        return (mean_return - self.risk_free_rate) / std_return
    
    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        
        Only considers negative returns for volatility calculation.
        """
        if len(returns) < 2:
            return 0.0
        
        # Downside returns only
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        # Annualize
        mean_return = returns.mean() * self.periods_per_year
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        
        return (mean_return - self.risk_free_rate) / downside_std
    
    def max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate Maximum Drawdown.
        
        Max Drawdown = (Peak - Trough) / Peak × 100
        
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if len(equity_curve) < 2:
            return 0.0, None, None
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_idx = equity_curve[:max_dd_idx].idxmax()
        
        return abs(max_dd), peak_idx, max_dd_idx
    
    def calmar_ratio(self, equity_curve: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar = Annualized Return / Maximum Drawdown
        """
        max_dd, _, _ = self.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        # Annualized return
        total_periods = len(returns)
        years = total_periods / self.periods_per_year
        total_ret = self.total_return(equity_curve)
        annualized_return = (1 + total_ret / 100) ** (1 / years) - 1 if years > 0 else 0
        
        return (annualized_return * 100) / max_dd
    
    def win_rate(self, trades: List[Trade]) -> float:
        """
        Calculate Win Rate.
        
        Win Rate = Winning Trades / Total Trades × 100
        """
        if not trades:
            return 0.0
        
        winning = sum(1 for t in trades if t.pnl and t.pnl > 0)
        return (winning / len(trades)) * 100
    
    def profit_factor(self, trades: List[Trade]) -> float:
        """
        Calculate Profit Factor.
        
        Profit Factor = Gross Profit / Gross Loss
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def avg_trade_duration(self, trades: List[Trade]) -> float:
        """
        Calculate Average Trade Duration in bars.
        """
        if not trades:
            return 0.0
        
        durations = [t.duration_bars for t in trades if t.duration_bars is not None]
        return np.mean(durations) if durations else 0.0
    
    def calculate_all_metrics(self, equity_curve: pd.Series, 
                               trades: List[Trade]) -> Dict:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: Series of portfolio values
            trades: List of completed trades
            
        Returns:
            Dictionary with all metrics
        """
        returns = self.calculate_returns(equity_curve)
        max_dd, peak_date, trough_date = self.max_drawdown(equity_curve)
        
        metrics = {
            # Return metrics
            'total_return_pct': round(self.total_return(equity_curve), 4),
            'total_return_points': round(equity_curve.iloc[-1] - equity_curve.iloc[0], 2) if len(equity_curve) > 0 else 0,
            
            # Risk-adjusted metrics
            'sharpe_ratio': round(self.sharpe_ratio(returns), 4),
            'sortino_ratio': round(self.sortino_ratio(returns), 4),
            'calmar_ratio': round(self.calmar_ratio(equity_curve, returns), 4),
            
            # Drawdown metrics
            'max_drawdown_pct': round(max_dd, 4),
            'max_drawdown_peak': str(peak_date) if peak_date else None,
            'max_drawdown_trough': str(trough_date) if trough_date else None,
            
            # Trade metrics
            'total_trades': len(trades),
            'winning_trades': sum(1 for t in trades if t.pnl and t.pnl > 0),
            'losing_trades': sum(1 for t in trades if t.pnl and t.pnl < 0),
            'win_rate_pct': round(self.win_rate(trades), 2),
            'profit_factor': round(self.profit_factor(trades), 4),
            'avg_trade_duration_bars': round(self.avg_trade_duration(trades), 2),
            
            # P&L metrics
            'avg_win': round(np.mean([t.pnl for t in trades if t.pnl and t.pnl > 0]), 2) if any(t.pnl and t.pnl > 0 for t in trades) else 0,
            'avg_loss': round(np.mean([t.pnl for t in trades if t.pnl and t.pnl < 0]), 2) if any(t.pnl and t.pnl < 0 for t in trades) else 0,
            'max_win': round(max([t.pnl for t in trades if t.pnl], default=0), 2),
            'max_loss': round(min([t.pnl for t in trades if t.pnl], default=0), 2),
            
            # Long/Short breakdown
            'long_trades': sum(1 for t in trades if t.entry_type == 'LONG'),
            'short_trades': sum(1 for t in trades if t.entry_type == 'SHORT'),
        }
        
        return metrics


class Backtester:
    """
    Comprehensive backtester with train/test split.
    """
    
    def __init__(self, train_ratio: float = 0.7, initial_capital: float = 100000):
        """
        Initialize backtester.
        
        Args:
            train_ratio: Proportion of data for training (default: 70%)
            initial_capital: Starting capital for equity curve
        """
        self.train_ratio = train_ratio
        self.initial_capital = initial_capital
        self.metrics_calc = PerformanceMetrics()
        
        # Results storage
        self.train_result: Optional[BacktestResult] = None
        self.test_result: Optional[BacktestResult] = None
        self.full_result: Optional[BacktestResult] = None
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        n_train = int(len(df) * self.train_ratio)
        
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()
        
        return train_df, test_df
    
    def build_equity_curve(self, df: pd.DataFrame, trades: List[Trade]) -> pd.Series:
        """
        Build equity curve from trades.
        
        Args:
            df: DataFrame with timestamps
            trades: List of trades
            
        Returns:
            Series with equity values indexed by timestamp
        """
        # Start with initial capital
        equity = pd.Series(index=df['timestamp'], dtype=float)
        equity.iloc[0] = self.initial_capital
        
        # Track cumulative P&L
        cumulative_pnl = 0
        
        # Create trade lookup by exit time
        trade_pnl = {}
        for t in trades:
            if t.exit_time and t.pnl:
                trade_pnl[t.exit_time] = t.pnl
        
        # Build equity curve
        for i, ts in enumerate(df['timestamp']):
            if i == 0:
                continue
            
            # Add P&L if trade closed at this timestamp
            if ts in trade_pnl:
                cumulative_pnl += trade_pnl[ts]
            
            equity.iloc[i] = self.initial_capital + cumulative_pnl
        
        # Forward fill any NaN values
        equity = equity.ffill()
        
        return equity
    
    def run_backtest(self, df: pd.DataFrame, 
                     strategy: EMARegimeStrategy) -> Tuple[BacktestResult, BacktestResult, BacktestResult]:
        """
        Run backtest on train and test data.
        
        Args:
            df: Full dataset
            strategy: Trading strategy instance
            
        Returns:
            Tuple of (train_result, test_result, full_result)
        """
        # Split data
        train_df, test_df = self.split_data(df)
        
        print(f"Data Split:")
        print(f"  Training: {len(train_df)} bars ({self.train_ratio*100:.0f}%)")
        print(f"  Testing: {len(test_df)} bars ({(1-self.train_ratio)*100:.0f}%)")
        print(f"  Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        # Backtest on training data
        print("\nBacktesting on TRAINING data...")
        train_strategy = EMARegimeStrategy()
        train_signals, train_trades = train_strategy.backtest(train_df)
        train_equity = self.build_equity_curve(train_df, train_trades)
        train_metrics = self.metrics_calc.calculate_all_metrics(train_equity, train_trades)
        
        self.train_result = BacktestResult(
            period='train',
            start_date=train_df['timestamp'].min(),
            end_date=train_df['timestamp'].max(),
            trades=train_trades,
            equity_curve=train_equity,
            metrics=train_metrics
        )
        
        # Backtest on testing data
        print("Backtesting on TESTING data...")
        test_strategy = EMARegimeStrategy()
        test_signals, test_trades = test_strategy.backtest(test_df)
        test_equity = self.build_equity_curve(test_df, test_trades)
        test_metrics = self.metrics_calc.calculate_all_metrics(test_equity, test_trades)
        
        self.test_result = BacktestResult(
            period='test',
            start_date=test_df['timestamp'].min(),
            end_date=test_df['timestamp'].max(),
            trades=test_trades,
            equity_curve=test_equity,
            metrics=test_metrics
        )
        
        # Backtest on full data
        print("Backtesting on FULL data...")
        full_strategy = EMARegimeStrategy()
        full_signals, full_trades = full_strategy.backtest(df)
        full_equity = self.build_equity_curve(df, full_trades)
        full_metrics = self.metrics_calc.calculate_all_metrics(full_equity, full_trades)
        
        self.full_result = BacktestResult(
            period='full',
            start_date=df['timestamp'].min(),
            end_date=df['timestamp'].max(),
            trades=full_trades,
            equity_curve=full_equity,
            metrics=full_metrics
        )
        
        return self.train_result, self.test_result, self.full_result
    
    def print_results(self):
        """Print formatted backtest results."""
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        for result, name in [(self.train_result, 'TRAINING'), 
                              (self.test_result, 'TESTING'),
                              (self.full_result, 'FULL PERIOD')]:
            if result is None:
                continue
            
            m = result.metrics
            
            print(f"\n{'─' * 40}")
            print(f"{name} ({result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')})")
            print(f"{'─' * 40}")
            
            print(f"\n  Performance Metrics:")
            print(f"    Total Return:     {m['total_return_pct']:.2f}% ({m['total_return_points']:.2f} points)")
            print(f"    Sharpe Ratio:     {m['sharpe_ratio']:.4f}")
            print(f"    Sortino Ratio:    {m['sortino_ratio']:.4f}")
            print(f"    Calmar Ratio:     {m['calmar_ratio']:.4f}")
            print(f"    Max Drawdown:     {m['max_drawdown_pct']:.2f}%")
            
            print(f"\n  Trade Metrics:")
            print(f"    Total Trades:     {m['total_trades']}")
            print(f"    Win Rate:         {m['win_rate_pct']:.2f}%")
            print(f"    Profit Factor:    {m['profit_factor']:.4f}")
            print(f"    Avg Duration:     {m['avg_trade_duration_bars']:.1f} bars")
            
            print(f"\n  Trade Breakdown:")
            print(f"    Long Trades:      {m['long_trades']}")
            print(f"    Short Trades:     {m['short_trades']}")
            print(f"    Winning Trades:   {m['winning_trades']}")
            print(f"    Losing Trades:    {m['losing_trades']}")
            
            print(f"\n  P&L Statistics:")
            print(f"    Avg Win:          {m['avg_win']:.2f} points")
            print(f"    Avg Loss:         {m['avg_loss']:.2f} points")
            print(f"    Max Win:          {m['max_win']:.2f} points")
            print(f"    Max Loss:         {m['max_loss']:.2f} points")
    
    def generate_report(self) -> str:
        """Generate text report of backtest results."""
        lines = [
            "=" * 80,
            "BACKTEST REPORT",
            "=" * 80,
            "",
            f"Strategy: 5/15 EMA Crossover with Regime Filter",
            f"Train/Test Split: {self.train_ratio*100:.0f}% / {(1-self.train_ratio)*100:.0f}%",
            f"Initial Capital: {self.initial_capital:,.0f}",
            "",
        ]
        
        for result, name in [(self.train_result, 'TRAINING'), 
                              (self.test_result, 'TESTING'),
                              (self.full_result, 'FULL PERIOD')]:
            if result is None:
                continue
            
            m = result.metrics
            
            lines.extend([
                "=" * 80,
                f"{name} PERIOD",
                "=" * 80,
                f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
                "",
                "PERFORMANCE METRICS:",
                f"  Total Return:        {m['total_return_pct']:.4f}%",
                f"  Total Return Points: {m['total_return_points']:.2f}",
                f"  Sharpe Ratio:        {m['sharpe_ratio']:.4f}",
                f"  Sortino Ratio:       {m['sortino_ratio']:.4f}",
                f"  Calmar Ratio:        {m['calmar_ratio']:.4f}",
                f"  Max Drawdown:        {m['max_drawdown_pct']:.4f}%",
                "",
                "TRADE METRICS:",
                f"  Total Trades:        {m['total_trades']}",
                f"  Win Rate:            {m['win_rate_pct']:.2f}%",
                f"  Profit Factor:       {m['profit_factor']:.4f}",
                f"  Avg Trade Duration:  {m['avg_trade_duration_bars']:.2f} bars",
                "",
                "TRADE BREAKDOWN:",
                f"  Long Trades:         {m['long_trades']}",
                f"  Short Trades:        {m['short_trades']}",
                f"  Winning Trades:      {m['winning_trades']}",
                f"  Losing Trades:       {m['losing_trades']}",
                "",
                "P&L STATISTICS:",
                f"  Avg Win:             {m['avg_win']:.2f} points",
                f"  Avg Loss:            {m['avg_loss']:.2f} points",
                f"  Max Win:             {m['max_win']:.2f} points",
                f"  Max Loss:            {m['max_loss']:.2f} points",
                "",
            ])
        
        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])
        
        return "\n".join(lines)


def run_backtest(input_path: Path, output_dir: Path = None) -> Tuple[Backtester, Dict]:
    """
    Run complete backtest with train/test split.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (Backtester instance, summary dict)
    """
    print("=" * 80)
    print("BACKTESTING - Task 4.2")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    
    # Initialize backtester
    backtester = Backtester(train_ratio=0.7, initial_capital=100000)
    
    # Initialize strategy
    strategy = EMARegimeStrategy(fast_period=5, slow_period=15)
    
    # Run backtest
    print("\n" + "-" * 50)
    train_result, test_result, full_result = backtester.run_backtest(df, strategy)
    print("-" * 50)
    
    # Print results
    backtester.print_results()
    
    # Generate and save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = backtester.generate_report()
        report_path = output_dir / "backtest_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # Save equity curves
        equity_df = pd.DataFrame({
            'timestamp': full_result.equity_curve.index,
            'equity': full_result.equity_curve.values
        })
        equity_path = output_dir / "equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve saved to: {equity_path}")
    
    # Summary
    summary = {
        'train': train_result.metrics,
        'test': test_result.metrics,
        'full': full_result.metrics
    }
    
    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETE")
    print("=" * 80)
    
    return backtester, summary


def main():
    """Main function."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_dir = Path("data/processed")
    
    backtester, summary = run_backtest(input_path, output_dir)
    
    return backtester, summary


if __name__ == "__main__":
    main()
