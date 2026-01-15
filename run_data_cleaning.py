#!/usr/bin/env python3
"""
Main script to run data cleaning for Task 1.2

This script cleans NIFTY 50 Spot, Futures, and Options data
and generates a cleaning report.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_cleaner import main

if __name__ == "__main__":
    main()
