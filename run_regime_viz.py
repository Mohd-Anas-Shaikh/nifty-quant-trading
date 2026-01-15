#!/usr/bin/env python3
"""
Main script to create regime visualizations for Task 3.2

This script creates:
1. Price chart with regime overlay (color-coded)
2. Transition matrix heatmap
3. Regime statistics (IV, Greeks distribution per regime)
4. Duration histogram
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.regime.regime_visualization import create_regime_visualizations

if __name__ == "__main__":
    input_path = Path("data/processed/nifty_features_5min.csv")
    paths = create_regime_visualizations(input_path)
    
    print("\nGenerated visualizations:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
