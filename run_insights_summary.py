#!/usr/bin/env python3
"""
Main script to generate insights summary for Task 6.3

Answers key questions about outlier trades:
- What percentage are outliers?
- Average PnL comparison
- Regime patterns
- Time-of-day patterns
- IV characteristics
- Distinguishing features
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.insights_summary import run_insights_summary
import pickle

if __name__ == "__main__":
    outlier_analysis_path = Path("data/processed/outlier_analysis.pkl")
    
    insights = run_insights_summary(outlier_analysis_path)
    
    # Save insights
    output_path = Path("data/processed/insights_summary.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(insights, f)
    print(f"\nInsights saved to: {output_path}")
