#!/usr/bin/env python3
"""
Main script to run pattern recognition for Task 6.2

Compares outlier trades vs normal profitable trades using statistical tests.
Creates visualizations: scatter plot, box plots, correlation heatmap, time distribution.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.pattern_recognition import run_pattern_recognition
import pickle

if __name__ == "__main__":
    outlier_analysis_path = Path("data/processed/outlier_analysis.pkl")
    
    results = run_pattern_recognition(outlier_analysis_path)
    
    # Save results
    output_path = Path("data/processed/pattern_recognition.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")
