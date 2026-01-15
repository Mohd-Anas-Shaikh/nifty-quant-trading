"""
Hidden Markov Model Regime Detection - Task 3.1

This module implements HMM-based market regime classification:
- Regime +1: Uptrend
- Regime -1: Downtrend  
- Regime 0: Sideways

Input Features (Options-based):
- Average IV
- IV Spread
- PCR (OI-based)
- ATM Delta
- ATM Gamma
- ATM Vega
- Futures Basis
- Spot Returns

Uses hmmlearn library with 3 hidden states.
Training: First 70% of data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import warnings
import pickle

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    HMM assumes that the market is always in one of several hidden "states"
    (regimes), and we can only observe certain features that depend on
    these hidden states. The model learns:
    
    1. Transition probabilities: How likely is it to switch from one regime to another?
    2. Emission probabilities: What features do we expect to see in each regime?
    
    We then use these learned patterns to classify new data into regimes.
    """
    
    # Feature columns for HMM
    FEATURE_COLS = [
        'avg_iv',           # Average Implied Volatility
        'iv_spread',        # IV Spread (Call - Put)
        'pcr_oi',           # Put-Call Ratio (OI-based)
        'greeks_ce_atm_delta',  # ATM Call Delta
        'greeks_ce_atm_gamma',  # ATM Call Gamma
        'greeks_ce_atm_vega',   # ATM Call Vega
        'futures_basis_pct',    # Futures Basis (%)
        'spot_return_1'         # Spot Returns (1-period)
    ]
    
    # Regime labels
    REGIME_LABELS = {
        0: 'Sideways',
        1: 'Uptrend',
        -1: 'Downtrend'
    }
    
    def __init__(self, n_states: int = 3, n_iter: int = 100, random_state: int = 42):
        """
        Initialize HMM Regime Detector.
        
        Args:
            n_states: Number of hidden states (regimes)
            n_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Regime mapping (will be determined after training)
        self.regime_map = {}
        
        # Training info
        self.is_trained = False
        self.train_score = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for HMM training/prediction.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (filtered DataFrame, feature matrix)
        """
        # Filter rows with all required features
        df_filtered = df.dropna(subset=self.FEATURE_COLS).copy()
        
        if len(df_filtered) == 0:
            raise ValueError("No rows with complete feature data")
        
        # Extract feature matrix
        X = df_filtered[self.FEATURE_COLS].values
        
        return df_filtered, X
    
    def train(self, df: pd.DataFrame, train_ratio: float = 0.7) -> Dict:
        """
        Train HMM on the data.
        
        Args:
            df: DataFrame with features
            train_ratio: Proportion of data for training (default: 70%)
            
        Returns:
            Dictionary with training results
        """
        print("Preparing features for HMM training...")
        
        # Prepare features
        df_filtered, X = self.prepare_features(df)
        
        # Split into train/test
        n_train = int(len(X) * train_ratio)
        X_train = X[:n_train]
        X_test = X[n_train:]
        
        print(f"  Total samples with features: {len(X)}")
        print(f"  Training samples: {n_train} ({train_ratio*100:.0f}%)")
        print(f"  Test samples: {len(X_test)} ({(1-train_ratio)*100:.0f}%)")
        
        # Normalize features
        print("\nNormalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train HMM
        print(f"\nTraining HMM with {self.n_states} states...")
        self.model.fit(X_train_scaled)
        
        # Get training score (log-likelihood)
        self.train_score = self.model.score(X_train_scaled)
        test_score = self.model.score(X_test_scaled)
        
        print(f"  Training log-likelihood: {self.train_score:.2f}")
        print(f"  Test log-likelihood: {test_score:.2f}")
        
        # Predict states on training data
        train_states = self.model.predict(X_train_scaled)
        
        # Map states to regimes based on average returns in each state
        self._map_states_to_regimes(df_filtered.iloc[:n_train], train_states)
        
        self.is_trained = True
        
        # Compile results
        results = {
            'n_train': n_train,
            'n_test': len(X_test),
            'train_score': self.train_score,
            'test_score': test_score,
            'transition_matrix': self.model.transmat_,
            'regime_map': self.regime_map,
            'state_means': self.model.means_,
            'feature_names': self.FEATURE_COLS
        }
        
        return results
    
    def _map_states_to_regimes(self, df_train: pd.DataFrame, states: np.ndarray):
        """
        Map HMM states to meaningful regime labels based on returns.
        
        Strategy:
        - State with highest average return → Uptrend (+1)
        - State with lowest average return → Downtrend (-1)
        - Middle state → Sideways (0)
        """
        # Calculate average return for each state
        state_returns = {}
        
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0:
                avg_return = df_train.iloc[mask]['spot_return_1'].mean()
                state_returns[state] = avg_return
        
        # Sort states by average return
        sorted_states = sorted(state_returns.keys(), key=lambda x: state_returns[x])
        
        # Map: lowest return → -1, middle → 0, highest → +1
        if self.n_states == 3:
            self.regime_map = {
                sorted_states[0]: -1,  # Downtrend
                sorted_states[1]: 0,   # Sideways
                sorted_states[2]: 1    # Uptrend
            }
        else:
            # For other number of states, use simple mapping
            for i, state in enumerate(sorted_states):
                if i < len(sorted_states) // 3:
                    self.regime_map[state] = -1
                elif i >= 2 * len(sorted_states) // 3:
                    self.regime_map[state] = 1
                else:
                    self.regime_map[state] = 0
        
        print("\nRegime Mapping (based on average returns):")
        for state, regime in self.regime_map.items():
            regime_name = self.REGIME_LABELS[regime]
            avg_ret = state_returns[state]
            print(f"  State {state} → Regime {regime:+d} ({regime_name}), Avg Return: {avg_ret:.4f}%")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of regime labels (-1, 0, +1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        df_filtered, X = self.prepare_features(df)
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # Predict states
        states = self.model.predict(X_scaled)
        
        # Map to regimes
        regimes = np.array([self.regime_map[s] for s in states])
        
        return regimes, df_filtered.index
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get regime probabilities for new data.
        
        Returns:
            Array of shape (n_samples, 3) with probabilities for each regime
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        df_filtered, X = self.prepare_features(df)
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # Get state probabilities
        state_proba = self.model.predict_proba(X_scaled)
        
        # Reorder to regime order (-1, 0, +1)
        regime_proba = np.zeros((len(X), 3))
        for state, regime in self.regime_map.items():
            regime_idx = regime + 1  # -1→0, 0→1, +1→2
            regime_proba[:, regime_idx] = state_proba[:, state]
        
        return regime_proba, df_filtered.index
    
    def add_regime_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime predictions to the full DataFrame.
        
        Adds columns:
        - regime: Predicted regime (-1, 0, +1)
        - regime_label: Text label (Downtrend, Sideways, Uptrend)
        - regime_prob_down: Probability of downtrend
        - regime_prob_side: Probability of sideways
        - regime_prob_up: Probability of uptrend
        """
        df = df.copy()
        
        # Initialize columns with NaN
        df['regime'] = np.nan
        df['regime_label'] = ''
        df['regime_prob_down'] = np.nan
        df['regime_prob_side'] = np.nan
        df['regime_prob_up'] = np.nan
        
        # Predict regimes
        regimes, valid_idx = self.predict(df)
        proba, _ = self.predict_proba(df)
        
        # Fill in predictions
        df.loc[valid_idx, 'regime'] = regimes
        df.loc[valid_idx, 'regime_label'] = [self.REGIME_LABELS[r] for r in regimes]
        df.loc[valid_idx, 'regime_prob_down'] = proba[:, 0]
        df.loc[valid_idx, 'regime_prob_side'] = proba[:, 1]
        df.loc[valid_idx, 'regime_prob_up'] = proba[:, 2]
        
        return df
    
    def get_regime_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about regime distribution."""
        if 'regime' not in df.columns:
            df = self.add_regime_to_dataframe(df)
        
        regime_counts = df['regime'].value_counts()
        total = regime_counts.sum()
        
        stats = {
            'total_classified': int(total),
            'regime_distribution': {}
        }
        
        for regime in [-1, 0, 1]:
            count = regime_counts.get(regime, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.REGIME_LABELS[regime]
            stats['regime_distribution'][label] = {
                'count': int(count),
                'percentage': round(pct, 2)
            }
        
        return stats
    
    def save_model(self, path: Path):
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'regime_map': self.regime_map,
            'is_trained': self.is_trained,
            'train_score': self.train_score,
            'feature_cols': self.FEATURE_COLS
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: Path):
        """Load trained model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.regime_map = model_data['regime_map']
        self.is_trained = model_data['is_trained']
        self.train_score = model_data['train_score']
        print(f"Model loaded from: {path}")


def train_hmm_regime_detector(input_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Train HMM regime detector and add predictions to dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output (optional)
        
    Returns:
        Tuple of (DataFrame with regimes, training results)
    """
    print("=" * 70)
    print("HMM REGIME DETECTION - Task 3.1")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Total records: {len(df)}")
    
    # Initialize HMM detector
    print("\nInitializing HMM with 3 states...")
    detector = HMMRegimeDetector(n_states=3, n_iter=100, random_state=42)
    
    # Print feature info
    print("\nInput Features:")
    for feat in detector.FEATURE_COLS:
        print(f"  - {feat}")
    
    # Train model
    print("\n" + "-" * 50)
    results = detector.train(df, train_ratio=0.7)
    print("-" * 50)
    
    # Print transition matrix
    print("\nTransition Matrix (probability of switching regimes):")
    trans_mat = results['transition_matrix']
    print("         To:")
    print("From:    ", end="")
    for i in range(3):
        print(f"State{i}  ", end="")
    print()
    for i in range(3):
        print(f"State{i}  ", end="")
        for j in range(3):
            print(f"{trans_mat[i,j]:.3f}   ", end="")
        print()
    
    # Add regimes to full dataset
    print("\nAdding regime predictions to dataset...")
    df = detector.add_regime_to_dataframe(df)
    
    # Get regime statistics
    stats = detector.get_regime_statistics(df)
    
    print("\nRegime Distribution:")
    for label, info in stats['regime_distribution'].items():
        print(f"  {label}: {info['count']} ({info['percentage']}%)")
    
    # Save model
    model_path = Path("data/processed/hmm_regime_model.pkl")
    detector.save_model(model_path)
    
    # Save dataset
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("HMM REGIME DETECTION COMPLETE")
    print("=" * 70)
    
    return df, results, stats


def main():
    """Main function to run HMM regime detection."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")
    
    df, results, stats = train_hmm_regime_detector(input_path, output_path)
    
    # Print sample
    print("\nSample data with regimes:")
    regime_cols = ['timestamp', 'spot_close', 'spot_return_1', 'avg_iv', 
                   'regime', 'regime_label', 'regime_prob_up', 'regime_prob_down']
    available = [c for c in regime_cols if c in df.columns]
    print(df[available].dropna().head(15))
    
    return df, results, stats


if __name__ == "__main__":
    main()
