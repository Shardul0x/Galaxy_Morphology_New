# data/loader.py
"""
Multi-format Data Loader for Astrophysics Data
Supports CSV, FITS, HDF5, and NumPy formats
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import os
from pathlib import Path

class AstrophysicsDataLoader:
    """
    Generic data loader for astrophysics datasets
    Supports multiple file formats commonly used in astronomy
    """
    
    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        dropna: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            feature_columns: List of column names to use as features
            target_columns: Optional target columns for supervised learning
            dropna: Whether to drop rows with missing values
            verbose: Print loading information
        """
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.dropna = dropna
        self.verbose = verbose
        
        self.data = None
        self.features = None
        self.targets = None
        self.metadata = {}
    
    def load_csv(
        self,
        filepath: str,
        sep: str = ',',
        comment: str = '#'
    ) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            sep: Column separator
            comment: Comment character
            
        Returns:
            DataFrame with loaded data
        """
        if self.verbose:
            print(f"Loading CSV file: {filepath}")
        
        df = pd.read_csv(filepath, sep=sep, comment=comment)
        
        if self.verbose:
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load_fits(self, filepath: str, hdu: int = 1) -> pd.DataFrame:
        """
        Load data from FITS file
        
        Args:
            filepath: Path to FITS file
            hdu: HDU (Header Data Unit) index to load
            
        Returns:
            DataFrame with loaded data
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy required for FITS files. Install with: pip install astropy")
        
        if self.verbose:
            print(f"Loading FITS file: {filepath}")
        
        with fits.open(filepath) as hdul:
            data = hdul[hdu].data
            
            # Convert FITS table to pandas DataFrame
            if data is not None:
                df = pd.DataFrame(np.array(data).byteswap().newbyteorder())
                
                if self.verbose:
                    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from HDU {hdu}")
                
                return df
            else:
                raise ValueError(f"No data found in HDU {hdu}")
    
    def load_hdf5(
        self,
        filepath: str,
        key: str = 'data',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from HDF5 file
        
        Args:
            filepath: Path to HDF5 file
            key: Key/path to dataset in HDF5 file
            columns: Optional list of columns to load
            
        Returns:
            DataFrame with loaded data
        """
        if self.verbose:
            print(f"Loading HDF5 file: {filepath}")
        
        df = pd.read_hdf(filepath, key=key, columns=columns)
        
        if self.verbose:
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load_numpy(self, filepath: str) -> pd.DataFrame:
        """
        Load data from NumPy file (.npy or .npz)
        
        Args:
            filepath: Path to NumPy file
            
        Returns:
            DataFrame with loaded data
        """
        if self.verbose:
            print(f"Loading NumPy file: {filepath}")
        
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            # If npz contains multiple arrays, concatenate them
            arrays = [data[key] for key in data.files]
            array = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
        else:
            array = np.load(filepath)
        
        df = pd.DataFrame(array)
        
        if self.verbose:
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load(
        self,
        filepath: str,
        file_format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Auto-detect format and load data
        
        Args:
            filepath: Path to data file
            file_format: Optional format specification ('csv', 'fits', 'hdf5', 'numpy')
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            DataFrame with loaded data
        """
        # Auto-detect format from extension
        if file_format is None:
            ext = Path(filepath).suffix.lower()
            format_map = {
                '.csv': 'csv',
                '.txt': 'csv',
                '.dat': 'csv',
                '.fits': 'fits',
                '.fit': 'fits',
                '.h5': 'hdf5',
                '.hdf5': 'hdf5',
                '.npy': 'numpy',
                '.npz': 'numpy'
            }
            file_format = format_map.get(ext, 'csv')
        
        # Load based on format
        if file_format == 'csv':
            df = self.load_csv(filepath, **kwargs)
        elif file_format == 'fits':
            df = self.load_fits(filepath, **kwargs)
        elif file_format == 'hdf5':
            df = self.load_hdf5(filepath, **kwargs)
        elif file_format == 'numpy':
            df = self.load_numpy(filepath)
        else:
            raise ValueError(f"Unknown format: {file_format}")
        
        # Store data
        self.data = df
        self._process_data()
        
        return df
    
    def _process_data(self):
        """Process loaded data: select features, handle missing values"""
        if self.data is None:
            return
        
        df = self.data.copy()
        
        # Drop missing values if requested
        if self.dropna:
            initial_len = len(df)
            df = df.dropna()
            if self.verbose and len(df) < initial_len:
                print(f"  Dropped {initial_len - len(df)} rows with missing values")
        
        # Select feature columns
        if self.feature_columns:
            available_cols = [col for col in self.feature_columns if col in df.columns]
            if len(available_cols) < len(self.feature_columns):
                missing = set(self.feature_columns) - set(available_cols)
                print(f"  Warning: Missing columns: {missing}")
            
            self.features = df[available_cols].values
            
            if self.verbose:
                print(f"  Selected {len(available_cols)} feature columns")
        else:
            self.features = df.values
        
        # Select target columns if specified
        if self.target_columns:
            available_targets = [col for col in self.target_columns if col in df.columns]
            if available_targets:
                self.targets = df[available_targets].values
                if self.verbose:
                    print(f"  Selected {len(available_targets)} target columns")
        
        # Store metadata
        self.metadata = {
            'n_samples': len(df),
            'n_features': self.features.shape[1] if self.features is not None else 0,
            'feature_names': self.feature_columns if self.feature_columns else list(df.columns),
            'shape': df.shape
        }
    
    def get_features(self) -> np.ndarray:
        """Get feature array"""
        if self.features is None:
            raise ValueError("No data loaded. Call load() first.")
        return self.features
    
    def get_targets(self) -> Optional[np.ndarray]:
        """Get target array if available"""
        return self.targets
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get original DataFrame"""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        return self.data
    
    def split_train_test(
        self,
        test_size: float = 0.2,
        random_state: Optional[int] = 42,
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Split data into train and test sets
        
        Args:
            test_size: Fraction of data for test set
            random_state: Random seed
            shuffle: Whether to shuffle before splitting
            
        Returns:
            X_train, X_test, y_train, y_test (y values are None if no targets)
        """
        from sklearn.model_selection import train_test_split
        
        X = self.get_features()
        y = self.get_targets()
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            y_train, y_test = None, None
        
        if self.verbose:
            print(f"\nData split:")
            print(f"  Train: {len(X_train)} samples")
            print(f"  Test:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


# Example loader configurations for common datasets
class GalaxyZooLoader(AstrophysicsDataLoader):
    """Specialized loader for Galaxy Zoo dataset"""
    
    def __init__(self, **kwargs):
        feature_columns = [
            'P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
            'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED'
        ]
        super().__init__(feature_columns=feature_columns, **kwargs)


class StellarParametersLoader(AstrophysicsDataLoader):
    """Specialized loader for stellar parameter catalogs"""
    
    def __init__(self, **kwargs):
        feature_columns = [
            'mass', 'luminosity', 'temperature', 'radius', 'metallicity', 'age'
        ]
        super().__init__(feature_columns=feature_columns, **kwargs)


class ExoplanetLoader(AstrophysicsDataLoader):
    """Specialized loader for exoplanet catalogs"""
    
    def __init__(self, **kwargs):
        feature_columns = [
            'period', 'depth', 'duration', 'impact_parameter',
            'planet_radius', 'stellar_radius'
        ]
        super().__init__(feature_columns=feature_columns, **kwargs)
