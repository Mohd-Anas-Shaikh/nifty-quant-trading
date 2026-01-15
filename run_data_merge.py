#!/usr/bin/env python3
"""
Main script to run data merging for Task 1.3

This script merges NIFTY 50 Spot, Futures, and Options data
into a single unified dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_merger import main

if __name__ == "__main__":
    main()
