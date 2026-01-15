#!/usr/bin/env python3
"""
Main script to train ML models for Task 5.2

This script trains:
- Model A: XGBoost with time-series cross-validation
- Model B: LSTM with sequence input (last 10 candles)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
from src.ml.model_training import train_models

if __name__ == "__main__":
    # Load ML dataset
    data_path = Path("data/processed/ml_dataset.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    output_dir = Path("data/processed")
    
    results = train_models(data, output_dir)
