#!/usr/bin/env python3
"""
Main script to run data acquisition for Task 1.1

This script fetches NIFTY 50 Spot, Futures, and Options data
and saves them to CSV files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.nse_data_fetcher import main

if __name__ == "__main__":
    main()
