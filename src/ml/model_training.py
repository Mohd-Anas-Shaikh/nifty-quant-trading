"""
Model Training Module - Task 5.2

Train two models for trade profitability prediction:

Model A: XGBoost
- Gradient boosting classifier
- Time-series cross-validation

Model B: LSTM
- Input: Sequence of last 10 candles
- Architecture: LSTM → Dropout → Dense → Output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

import xgboost as xgb

warnings.filterwarnings('ignore')

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM model will be skipped.")


class XGBoostModel:
    """
    XGBoost Gradient Boosting Classifier with Time-Series Cross-Validation.
    
    XGBoost is an optimized gradient boosting algorithm that:
    - Builds trees sequentially, each correcting errors of previous trees
    - Handles missing values automatically
    - Includes regularization to prevent overfitting
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            n_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.cv_results = []
        self.feature_importance = None
        
        # XGBoost parameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': random_state,
            'use_label_encoder': False,
            'verbosity': 0
        }
    
    def time_series_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform time-series cross-validation.
        
        Time-series CV ensures:
        - Training data always comes before test data
        - No lookahead bias
        - Realistic evaluation of model performance
        """
        print(f"\nPerforming {self.n_splits}-fold Time-Series Cross-Validation...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train_cv, y_train_cv, verbose=False)
            
            # Predict
            y_pred = model.predict(X_val_cv)
            y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
            
            # Calculate metrics
            cv_metrics['accuracy'].append(accuracy_score(y_val_cv, y_pred))
            cv_metrics['precision'].append(precision_score(y_val_cv, y_pred, zero_division=0))
            cv_metrics['recall'].append(recall_score(y_val_cv, y_pred, zero_division=0))
            cv_metrics['f1'].append(f1_score(y_val_cv, y_pred, zero_division=0))
            cv_metrics['auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
            
            print(f"  Fold {fold+1}: Accuracy={cv_metrics['accuracy'][-1]:.4f}, "
                  f"AUC={cv_metrics['auc'][-1]:.4f}")
        
        # Calculate mean and std
        cv_summary = {}
        for metric, values in cv_metrics.items():
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)
        
        self.cv_results = cv_summary
        
        print(f"\nCV Results (Mean ± Std):")
        print(f"  Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
        print(f"  Precision: {cv_summary['precision_mean']:.4f} ± {cv_summary['precision_std']:.4f}")
        print(f"  Recall: {cv_summary['recall_mean']:.4f} ± {cv_summary['recall_std']:.4f}")
        print(f"  F1 Score: {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
        print(f"  AUC: {cv_summary['auc_mean']:.4f} ± {cv_summary['auc_std']:.4f}")
        
        return cv_summary
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'XGBoostModel':
        """
        Train the final XGBoost model on full training data.
        """
        print("\nTraining final XGBoost model...")
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, verbose=False)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Model trained on {len(X_train)} samples")
        print(f"  Top 5 features:")
        for _, row in self.feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test data."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def save(self, path: Path):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'params': self.params,
            'cv_results': self.cv_results,
            'feature_importance': self.feature_importance
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"XGBoost model saved to: {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.params = model_data['params']
        self.cv_results = model_data['cv_results']
        self.feature_importance = model_data['feature_importance']


class LSTMModel:
    """
    LSTM Neural Network for Sequence-based Prediction.
    
    LSTM (Long Short-Term Memory) is a type of recurrent neural network that:
    - Processes sequential data (time series)
    - Remembers long-term dependencies
    - Uses gates to control information flow
    
    Architecture:
    Input (sequence of 10 candles) → LSTM → Dropout → Dense → Output
    """
    
    def __init__(self, sequence_length: int = 10, random_state: int = 42):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of past candles to use as input
            random_state: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds
        np.random.seed(random_state)
        if TF_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        For each sample, we use the previous `sequence_length` samples as input.
        """
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, n_features: int) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Architecture:
        - Input: (sequence_length, n_features)
        - LSTM layer with 64 units
        - Dropout (0.3) for regularization
        - Dense layer with 32 units
        - Output: 1 unit with sigmoid activation
        """
        model = Sequential([
            Input(shape=(self.sequence_length, n_features)),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> 'LSTMModel':
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM training.")
            return self
        
        print("\nPreparing sequences for LSTM...")
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        print(f"  Training sequences: {X_train_seq.shape}")
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            print(f"  Validation sequences: {X_val_seq.shape}")
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build model
        n_features = X_train_seq.shape[2]
        self.model = self.build_model(n_features)
        
        print("\nLSTM Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train
        print("\nTraining LSTM model...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            return np.array([])
        
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        return (y_pred_proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            return np.array([])
        
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test data."""
        if self.model is None:
            return {'error': 'Model not trained'}
        
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        y_pred_proba = self.model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test_seq, y_pred),
            'precision': precision_score(y_test_seq, y_pred, zero_division=0),
            'recall': recall_score(y_test_seq, y_pred, zero_division=0),
            'f1': f1_score(y_test_seq, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test_seq, y_pred_proba) if len(np.unique(y_test_seq)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_test_seq, y_pred).tolist()
        }
        
        return metrics
    
    def save(self, path: Path):
        """Save model to file."""
        if self.model is None:
            print("No model to save.")
            return
        
        # Save Keras model
        model_path = path.with_suffix('.keras')
        self.model.save(model_path)
        
        # Save metadata
        meta_path = path.with_suffix('.pkl')
        meta = {
            'sequence_length': self.sequence_length,
            'history': self.history.history if self.history else None
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"LSTM model saved to: {model_path}")
    
    def load(self, path: Path):
        """Load model from file."""
        model_path = path.with_suffix('.keras')
        self.model = keras.models.load_model(model_path)
        
        meta_path = path.with_suffix('.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.sequence_length = meta['sequence_length']


def train_models(data: Dict, output_dir: Path = None) -> Dict:
    """
    Train both XGBoost and LSTM models.
    
    Args:
        data: ML dataset dictionary from problem_definition
        output_dir: Directory to save models
        
    Returns:
        Dictionary with trained models and results
    """
    print("=" * 70)
    print("MODEL TRAINING - Task 5.2")
    print("=" * 70)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    results = {}
    
    # =========================================
    # Model A: XGBoost
    # =========================================
    print("\n" + "=" * 50)
    print("MODEL A: XGBoost")
    print("=" * 50)
    
    xgb_model = XGBoostModel(n_splits=5, random_state=42)
    
    # Time-series cross-validation
    cv_results = xgb_model.time_series_cv(X_train, y_train)
    
    # Train final model
    xgb_model.train(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    xgb_test_metrics = xgb_model.evaluate(X_test, y_test)
    
    print(f"\nXGBoost Test Results:")
    print(f"  Accuracy: {xgb_test_metrics['accuracy']:.4f}")
    print(f"  Precision: {xgb_test_metrics['precision']:.4f}")
    print(f"  Recall: {xgb_test_metrics['recall']:.4f}")
    print(f"  F1 Score: {xgb_test_metrics['f1']:.4f}")
    print(f"  AUC: {xgb_test_metrics['auc']:.4f}")
    
    results['xgboost'] = {
        'model': xgb_model,
        'cv_results': cv_results,
        'test_metrics': xgb_test_metrics,
        'feature_importance': xgb_model.feature_importance
    }
    
    # Save XGBoost model
    if output_dir:
        xgb_model.save(output_dir / 'xgboost_model.pkl')
    
    # =========================================
    # Model B: LSTM
    # =========================================
    print("\n" + "=" * 50)
    print("MODEL B: LSTM")
    print("=" * 50)
    
    if TF_AVAILABLE:
        lstm_model = LSTMModel(sequence_length=10, random_state=42)
        
        # Convert to numpy arrays
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # Train LSTM
        lstm_model.train(
            X_train_np, y_train_np,
            X_val=X_test_np, y_val=y_test_np,
            epochs=50, batch_size=16
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        lstm_test_metrics = lstm_model.evaluate(X_test_np, y_test_np)
        
        print(f"\nLSTM Test Results:")
        print(f"  Accuracy: {lstm_test_metrics['accuracy']:.4f}")
        print(f"  Precision: {lstm_test_metrics['precision']:.4f}")
        print(f"  Recall: {lstm_test_metrics['recall']:.4f}")
        print(f"  F1 Score: {lstm_test_metrics['f1']:.4f}")
        print(f"  AUC: {lstm_test_metrics['auc']:.4f}")
        
        results['lstm'] = {
            'model': lstm_model,
            'test_metrics': lstm_test_metrics,
            'history': lstm_model.history.history if lstm_model.history else None
        }
        
        # Save LSTM model
        if output_dir:
            lstm_model.save(output_dir / 'lstm_model')
    else:
        print("TensorFlow not available. Skipping LSTM model.")
        results['lstm'] = {'error': 'TensorFlow not available'}
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    
    print("\nModel Comparison (Test Set):")
    print(f"{'Metric':<15} {'XGBoost':<15} {'LSTM':<15}")
    print("-" * 45)
    
    xgb_m = results['xgboost']['test_metrics']
    lstm_m = results.get('lstm', {}).get('test_metrics', {})
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        xgb_val = f"{xgb_m.get(metric, 0):.4f}"
        lstm_val = f"{lstm_m.get(metric, 0):.4f}" if lstm_m and metric in lstm_m else "N/A"
        print(f"{metric:<15} {xgb_val:<15} {lstm_val:<15}")
    
    return results


def main():
    """Main function."""
    # Load ML dataset
    data_path = Path("data/processed/ml_dataset.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    output_dir = Path("data/processed")
    
    results = train_models(data, output_dir)
    
    return results


if __name__ == "__main__":
    main()
