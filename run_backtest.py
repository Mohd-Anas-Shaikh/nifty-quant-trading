#!/usr/bin/env python3
"""
Main script to run backtesting for Task 4.2

This script runs comprehensive backtesting with:
- Train/Test split (70/30)
- Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown
- Trade metrics: Win Rate, Profit Factor, Avg Duration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.backtester import run_backtest

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_dir = Path("data/processed")
    
    backtester, summary = run_backtest(input_path, output_dir)
