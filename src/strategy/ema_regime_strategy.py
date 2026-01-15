"""
EMA Regime Strategy Module - Task 4.1

Implements 5/15 EMA crossover strategy with regime filter:

LONG Entry:
- 5 EMA crosses above 15 EMA
- Current regime = +1 (Uptrend)
- Enter at next candle open

LONG Exit:
- 5 EMA crosses below 15 EMA
- Exit at next candle open

SHORT Entry:
- 5 EMA crosses below 15 EMA
- Current regime = -1 (Downtrend)
- Enter at next candle open

SHORT Exit:
- 5 EMA crosses above 15 EMA
- Exit at next candle open

No trades in Regime 0 (Sideways).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class Position(Enum):
    """Position types."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    entry_type: str  # 'LONG' or 'SHORT'
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    regime_at_entry: Optional[int] = None
    duration_bars: Optional[int] = None
    
    def close(self, exit_time: pd.Timestamp, exit_price: float, duration: int):
        """Close the trade and calculate P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.duration_bars = duration
        
        if self.entry_type == 'LONG':
            self.pnl = exit_price - self.entry_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100
        else:  # SHORT
            self.pnl = self.entry_price - exit_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100


class EMARegimeStrategy:
    """
    5/15 EMA Crossover Strategy with Regime Filter.
    
    This strategy combines:
    1. EMA crossover signals for entry/exit timing
    2. HMM regime filter to trade only in trending markets
    
    Key Rules:
    - Only LONG in Uptrend (regime +1)
    - Only SHORT in Downtrend (regime -1)
    - No trades in Sideways (regime 0)
    - Enter at next candle open after signal
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 15):
        """
        Initialize strategy.
        
        Args:
            fast_period: Fast EMA period (default: 5)
            slow_period: Slow EMA period (default: 15)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Trading state
        self.position = Position.FLAT
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        
        # Statistics
        self.stats = {}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on EMA crossover and regime.
        
        Signal Logic:
        - signal = 1: LONG entry signal
        - signal = -1: SHORT entry signal
        - signal = 0: No signal or exit
        
        Args:
            df: DataFrame with ema_fast, ema_slow, ema_crossover, regime columns
            
        Returns:
            DataFrame with signal columns added
        """
        df = df.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_type'] = ''
        df['position'] = 0
        
        # Forward fill regime to all rows (regime is only available for hourly data)
        df['regime_filled'] = df['regime'].ffill()
        
        # EMA crossover signals (already calculated in Task 2.1)
        # ema_crossover: 1 = bullish (fast crosses above slow)
        #               -1 = bearish (fast crosses below slow)
        
        # Generate entry signals with regime filter
        # LONG: Bullish crossover + Uptrend regime
        long_entry = (df['ema_crossover'] == 1) & (df['regime_filled'] == 1)
        
        # SHORT: Bearish crossover + Downtrend regime
        short_entry = (df['ema_crossover'] == -1) & (df['regime_filled'] == -1)
        
        # Exit signals (crossover in opposite direction)
        long_exit = df['ema_crossover'] == -1
        short_exit = df['ema_crossover'] == 1
        
        # Mark signals using np.select for vectorized assignment
        df['signal'] = 0
        df.loc[long_entry, 'signal'] = 1
        df.loc[short_entry, 'signal'] = -1
        
        # Signal types using np.select
        conditions = [
            long_entry,
            short_entry,
            long_exit & ~long_entry,
            short_exit & ~short_entry
        ]
        choices = ['LONG_ENTRY', 'SHORT_ENTRY', 'LONG_EXIT', 'SHORT_EXIT']
        df['signal_type'] = np.select(conditions, choices, default='')
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Trade]]:
        """
        Run backtest on historical data.
        
        Entry is at NEXT candle open after signal.
        Exit is at NEXT candle open after exit signal.
        
        Args:
            df: DataFrame with price and signal data
            
        Returns:
            Tuple of (DataFrame with positions, list of trades)
        """
        df = df.copy()
        
        # Generate signals first
        df = self.generate_signals(df)
        
        # Reset state
        self.position = Position.FLAT
        self.current_trade = None
        self.trades = []
        
        # Track position for each bar
        positions = []
        trade_ids = []
        
        # Iterate through data
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Get next candle open price (for entry/exit)
            if i + 1 < len(df):
                next_open = df.iloc[i + 1]['spot_open']
                next_time = df.iloc[i + 1]['timestamp']
            else:
                next_open = row['spot_close']
                next_time = row['timestamp']
            
            # Current signal
            signal_type = row['signal_type']
            
            # Process signals
            if self.position == Position.FLAT:
                # Look for entry signals
                if signal_type == 'LONG_ENTRY':
                    # Enter LONG at next candle open
                    self.position = Position.LONG
                    self.current_trade = Trade(
                        entry_time=next_time,
                        entry_price=next_open,
                        entry_type='LONG',
                        regime_at_entry=row['regime_filled']
                    )
                    
                elif signal_type == 'SHORT_ENTRY':
                    # Enter SHORT at next candle open
                    self.position = Position.SHORT
                    self.current_trade = Trade(
                        entry_time=next_time,
                        entry_price=next_open,
                        entry_type='SHORT',
                        regime_at_entry=row['regime_filled']
                    )
            
            elif self.position == Position.LONG:
                # Look for exit signals
                if signal_type in ['LONG_EXIT', 'SHORT_ENTRY']:
                    # Exit LONG at next candle open
                    if self.current_trade:
                        duration = i - df.index.get_loc(
                            df[df['timestamp'] == self.current_trade.entry_time].index[0]
                        ) if self.current_trade.entry_time in df['timestamp'].values else 0
                        
                        self.current_trade.close(next_time, next_open, duration)
                        self.trades.append(self.current_trade)
                    
                    self.position = Position.FLAT
                    self.current_trade = None
                    
                    # If SHORT_ENTRY, also enter short
                    if signal_type == 'SHORT_ENTRY':
                        self.position = Position.SHORT
                        self.current_trade = Trade(
                            entry_time=next_time,
                            entry_price=next_open,
                            entry_type='SHORT',
                            regime_at_entry=row['regime_filled']
                        )
            
            elif self.position == Position.SHORT:
                # Look for exit signals
                if signal_type in ['SHORT_EXIT', 'LONG_ENTRY']:
                    # Exit SHORT at next candle open
                    if self.current_trade:
                        duration = i - df.index.get_loc(
                            df[df['timestamp'] == self.current_trade.entry_time].index[0]
                        ) if self.current_trade.entry_time in df['timestamp'].values else 0
                        
                        self.current_trade.close(next_time, next_open, duration)
                        self.trades.append(self.current_trade)
                    
                    self.position = Position.FLAT
                    self.current_trade = None
                    
                    # If LONG_ENTRY, also enter long
                    if signal_type == 'LONG_ENTRY':
                        self.position = Position.LONG
                        self.current_trade = Trade(
                            entry_time=next_time,
                            entry_price=next_open,
                            entry_type='LONG',
                            regime_at_entry=row['regime_filled']
                        )
            
            # Record position
            positions.append(self.position.value)
            trade_ids.append(len(self.trades) if self.current_trade else len(self.trades))
        
        # Close any open trade at end
        if self.current_trade:
            last_row = df.iloc[-1]
            self.current_trade.close(
                last_row['timestamp'],
                last_row['spot_close'],
                len(df) - 1
            )
            self.trades.append(self.current_trade)
        
        df['position'] = positions
        df['trade_id'] = trade_ids
        
        return df, self.trades
    
    def calculate_statistics(self, trades: List[Trade] = None) -> Dict:
        """
        Calculate trading statistics.
        
        Args:
            trades: List of trades (uses self.trades if None)
            
        Returns:
            Dictionary with statistics
        """
        trades = trades or self.trades
        
        if not trades:
            return {'error': 'No trades to analyze'}
        
        # Basic counts
        total_trades = len(trades)
        long_trades = [t for t in trades if t.entry_type == 'LONG']
        short_trades = [t for t in trades if t.entry_type == 'SHORT']
        
        # P&L analysis
        pnls = [t.pnl for t in trades if t.pnl is not None]
        pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        # Calculate statistics
        stats = {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
            'total_pnl': sum(pnls) if pnls else 0,
            'total_pnl_pct': sum(pnl_pcts) if pnl_pcts else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
        }
        
        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades if t.pnl])
        gross_loss = abs(sum([t.pnl for t in losing_trades if t.pnl]))
        stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duration statistics
        durations = [t.duration_bars for t in trades if t.duration_bars is not None]
        if durations:
            stats['avg_duration_bars'] = np.mean(durations)
            stats['max_duration_bars'] = max(durations)
            stats['min_duration_bars'] = min(durations)
        
        # Long vs Short performance
        long_pnls = [t.pnl for t in long_trades if t.pnl is not None]
        short_pnls = [t.pnl for t in short_trades if t.pnl is not None]
        
        stats['long_total_pnl'] = sum(long_pnls) if long_pnls else 0
        stats['short_total_pnl'] = sum(short_pnls) if short_pnls else 0
        stats['long_win_rate'] = len([t for t in long_trades if t.pnl and t.pnl > 0]) / len(long_trades) * 100 if long_trades else 0
        stats['short_win_rate'] = len([t for t in short_trades if t.pnl and t.pnl > 0]) / len(short_trades) * 100 if short_trades else 0
        
        self.stats = stats
        return stats
    
    def get_trade_log(self, trades: List[Trade] = None) -> pd.DataFrame:
        """
        Convert trades to DataFrame for analysis.
        
        Args:
            trades: List of trades
            
        Returns:
            DataFrame with trade details
        """
        trades = trades or self.trades
        
        if not trades:
            return pd.DataFrame()
        
        trade_data = []
        for i, t in enumerate(trades):
            trade_data.append({
                'trade_id': i + 1,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'type': t.entry_type,
                'regime': t.regime_at_entry,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration_bars': t.duration_bars
            })
        
        return pd.DataFrame(trade_data)


def run_strategy(input_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Run the EMA Regime strategy on data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output (optional)
        
    Returns:
        Tuple of (DataFrame with signals, statistics dict, trade log DataFrame)
    """
    print("=" * 70)
    print("EMA REGIME STRATEGY - Task 4.1")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    
    # Initialize strategy
    print("\nInitializing 5/15 EMA Strategy with Regime Filter...")
    strategy = EMARegimeStrategy(fast_period=5, slow_period=15)
    
    # Print strategy rules
    print("\nStrategy Rules:")
    print("  LONG Entry: EMA(5) crosses above EMA(15) + Regime = +1")
    print("  LONG Exit:  EMA(5) crosses below EMA(15)")
    print("  SHORT Entry: EMA(5) crosses below EMA(15) + Regime = -1")
    print("  SHORT Exit:  EMA(5) crosses above EMA(15)")
    print("  No trades in Regime 0 (Sideways)")
    
    # Run backtest
    print("\nRunning backtest...")
    df_signals, trades = strategy.backtest(df)
    
    # Calculate statistics
    stats = strategy.calculate_statistics(trades)
    
    # Get trade log
    trade_log = strategy.get_trade_log(trades)
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\nTrade Summary:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Long Trades: {stats['long_trades']}")
    print(f"  Short Trades: {stats['short_trades']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades: {stats['losing_trades']}")
    
    print(f"\nPerformance:")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Total P&L: {stats['total_pnl']:.2f} points")
    print(f"  Total P&L %: {stats['total_pnl_pct']:.2f}%")
    print(f"  Avg P&L per Trade: {stats['avg_pnl']:.2f} points")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    print(f"\nBest/Worst:")
    print(f"  Max Win: {stats['max_win']:.2f} points")
    print(f"  Max Loss: {stats['max_loss']:.2f} points")
    print(f"  Avg Win: {stats['avg_win']:.2f} points")
    print(f"  Avg Loss: {stats['avg_loss']:.2f} points")
    
    print(f"\nLong vs Short:")
    print(f"  Long P&L: {stats['long_total_pnl']:.2f} points (Win Rate: {stats['long_win_rate']:.2f}%)")
    print(f"  Short P&L: {stats['short_total_pnl']:.2f} points (Win Rate: {stats['short_win_rate']:.2f}%)")
    
    # Save outputs
    if output_path:
        df_signals.to_csv(output_path, index=False)
        print(f"\nSaved signals to: {output_path}")
        
        # Save trade log
        trade_log_path = output_path.parent / "trade_log.csv"
        trade_log.to_csv(trade_log_path, index=False)
        print(f"Saved trade log to: {trade_log_path}")
    
    print("\n" + "=" * 70)
    print("STRATEGY IMPLEMENTATION COMPLETE")
    print("=" * 70)
    
    return df_signals, stats, trade_log


def main():
    """Main function."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, stats, trade_log = run_strategy(input_path, output_path)
    
    # Print sample trades
    print("\nSample Trades:")
    print(trade_log.head(10))
    
    return df, stats, trade_log


if __name__ == "__main__":
    main()
