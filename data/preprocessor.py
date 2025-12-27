# data/preprocessor.py
"""
Data Preprocessing for Astrophysics Data Augmentation
Handles scaling, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from typing import Optional, List, Tuple, Dict, Any, Union  # ← Add Union here
import pickle


class AstrophysicsPreprocessor:
    """
    Preprocessing pipeline for astrophysics data
    Handles various scaling and transformation methods
    """
    
    def __init__(
        self,
        scaler_type: str = 'minmax',
        feature_ranges: Optional[dict] = None,
        log_transform_features: Optional[List[int]] = None,
        clip_outliers: bool = False,
        outlier_quantiles: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust', 'power', 'quantile')
            feature_ranges: Optional dict of physical ranges for each feature
            log_transform_features: Indices of features to log-transform
            clip_outliers: Whether to clip outliers
            outlier_quantiles: Quantile range for clipping
        """
        self.scaler_type = scaler_type
        self.feature_ranges = feature_ranges
        self.log_transform_features = log_transform_features or []
        self.clip_outliers = clip_outliers
        self.outlier_quantiles = outlier_quantiles
        
        # Initialize scaler
        self.scaler = self._create_scaler(scaler_type)
        self.is_fitted = False
        
        # Store statistics
        self.clip_bounds = None
        self.feature_stats = {}
    
    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type"""
        scalers = {
            'minmax': MinMaxScaler(feature_range=(0, 1)),
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson'),
            'quantile': QuantileTransformer(output_distribution='uniform')
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Choose from {list(scalers.keys())}")
        
        return scalers[scaler_type]
    
    def fit(self, X: np.ndarray) -> 'AstrophysicsPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            X: Training data [n_samples, n_features]
            
        Returns:
            self
        """
        X = X.copy()
        
        # Apply log transform if specified
        if self.log_transform_features:
            for idx in self.log_transform_features:
                # Add small epsilon to avoid log(0)
                X[:, idx] = np.log1p(np.abs(X[:, idx]))
        
        # Compute outlier clipping bounds
        if self.clip_outliers:
            self.clip_bounds = {}
            for i in range(X.shape[1]):
                lower = np.quantile(X[:, i], self.outlier_quantiles[0])
                upper = np.quantile(X[:, i], self.outlier_quantiles[1])
                self.clip_bounds[i] = (lower, upper)
            
            # Apply clipping
            X = self._clip_outliers(X)
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Store feature statistics
        self.feature_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0)
        }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Data to transform [n_samples, n_features]
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        X = X.copy()
        
        # Apply log transform
        if self.log_transform_features:
            for idx in self.log_transform_features:
                X[:, idx] = np.log1p(np.abs(X[:, idx]))
        
        # Clip outliers
        if self.clip_outliers and self.clip_bounds:
            X = self._clip_outliers(X)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original space
        
        Args:
            X_scaled: Scaled data
            
        Returns:
            Data in original space
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        # Inverse scaling
        X = self.scaler.inverse_transform(X_scaled)
        
        # Inverse log transform
        if self.log_transform_features:
            for idx in self.log_transform_features:
                X[:, idx] = np.expm1(X[:, idx])
        
        return X
    
    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers based on stored bounds"""
        X_clipped = X.copy()
        for i, (lower, upper) in self.clip_bounds.items():
            X_clipped[:, i] = np.clip(X_clipped[:, i], lower, upper)
        return X_clipped
    
    def validate_physical_ranges(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Validate that features are within physical ranges
        
        Args:
            X: Data to validate
            feature_names: Optional list of feature names
            
        Returns:
            valid_mask: Boolean mask of valid samples
            invalid_indices: Indices of invalid samples
        """
        if self.feature_ranges is None:
            return np.ones(len(X), dtype=bool), []
        
        valid_mask = np.ones(len(X), dtype=bool)
        
        for i, (feature, (min_val, max_val)) in enumerate(self.feature_ranges.items()):
            if feature_names and feature in feature_names:
                idx = feature_names.index(feature)
            else:
                idx = i
            
            # Check if values are within range
            within_range = (X[:, idx] >= min_val) & (X[:, idx] <= max_val)
            valid_mask &= within_range
        
        invalid_indices = np.where(~valid_mask)[0].tolist()
        
        return valid_mask, invalid_indices
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'AstrophysicsPreprocessor':
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute feature importance based on variance
        
        Args:
            X: Input data
            
        Returns:
            Feature importance scores (normalized)
        """
        X_transformed = self.transform(X)
        variances = np.var(X_transformed, axis=0)
        importance = variances / np.sum(variances)
        return importance


class TimeSeriesPreprocessor:
    """
    Specialized preprocessor for time series astrophysics data
    Handles temporal features and sequence generation
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        stride: int = 1,
        normalize_by_initial: bool = False
    ):
        """
        Args:
            sequence_length: Length of sequences to generate
            stride: Step size between sequences
            normalize_by_initial: Normalize each sequence by its initial value
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize_by_initial = normalize_by_initial
    
    def create_sequences(
        self,
        X: np.ndarray,
        return_initial_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Create sequences from time series data
        
        Args:
            X: Time series data [n_timesteps, n_features]
            return_initial_states: Whether to return initial states separately
            
        Returns:
            sequences: Array of sequences [n_sequences, sequence_length, n_features]
            initial_states: Optional initial state for each sequence
        """
        n_timesteps, n_features = X.shape
        sequences = []
        initial_states = []
        
        for i in range(0, n_timesteps - self.sequence_length + 1, self.stride):
            sequence = X[i:i + self.sequence_length]
            sequences.append(sequence)
            
            if return_initial_states:
                initial_states.append(X[i])
        
        sequences = np.array(sequences)
        
        # Normalize by initial value if requested
        if self.normalize_by_initial:
            for i, seq in enumerate(sequences):
                initial = seq[0].copy()
                initial[initial == 0] = 1  # Avoid division by zero
                sequences[i] = seq / initial
        
        if return_initial_states:
            return sequences, np.array(initial_states)
        
        return sequences
    
    def generate_time_values(
        self,
        num_sequences: int,
        time_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Generate time values for sequences
        
        Args:
            num_sequences: Number of sequences
            time_range: (start_time, end_time)
            
        Returns:
            Time values [num_sequences, sequence_length]
        """
        t_values = np.linspace(
            time_range[0],
            time_range[1],
            self.sequence_length
        )
        
        # Repeat for each sequence
        time_array = np.tile(t_values, (num_sequences, 1))
        
        return time_array
