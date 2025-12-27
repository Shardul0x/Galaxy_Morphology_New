# config/model_config.py
"""
Model Configuration for VAE and PINN Hyperparameters
Provides presets and custom configuration options
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class VAEConfig:
    """Configuration for VAE model architecture and training"""
    
    # Architecture
    input_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    activation: str = 'leakyrelu'
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    beta: float = 1.0  # Beta for β-VAE
    beta_schedule: Optional[str] = None  # 'linear', 'cyclical', or None
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-5
    scheduler: Optional[str] = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', None
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'beta': self.beta,
            'beta_schedule': self.beta_schedule,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class PINNConfig:
    """Configuration for PINN model architecture and training"""
    
    # Architecture
    input_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = 'tanh'
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    physics_weight: float = 0.1  # Weight for physics loss
    
    # Time integration
    num_time_steps: int = 50
    time_range: tuple = field(default_factory=lambda: (0.0, 1.0))
    
    # Optimization
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    scheduler: Optional[str] = 'reduce_on_plateau'
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'physics_weight': self.physics_weight,
            'num_time_steps': self.num_time_steps,
            'time_range': self.time_range,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ===== PRESET CONFIGURATIONS =====

# Small model for testing
VAE_CONFIG_SMALL = VAEConfig(
    input_dim=9,
    hidden_dims=[32, 16],
    latent_dim=5,
    batch_size=32,
    num_epochs=50
)

PINN_CONFIG_SMALL = PINNConfig(
    input_dim=9,
    hidden_dims=[32, 32],
    batch_size=32,
    num_epochs=50
)

# Medium model for standard usage
VAE_CONFIG_MEDIUM = VAEConfig(
    input_dim=9,
    hidden_dims=[64, 32, 16],
    latent_dim=8,
    batch_size=64,
    num_epochs=100
)

PINN_CONFIG_MEDIUM = PINNConfig(
    input_dim=9,
    hidden_dims=[64, 64, 32],
    batch_size=64,
    num_epochs=100
)

# Large model for complex data
VAE_CONFIG_LARGE = VAEConfig(
    input_dim=9,
    hidden_dims=[128, 64, 32, 16],
    latent_dim=12,
    batch_size=128,
    num_epochs=200,
    dropout_rate=0.3
)

PINN_CONFIG_LARGE = PINNConfig(
    input_dim=9,
    hidden_dims=[128, 128, 64, 32],
    batch_size=128,
    num_epochs=200
)

# Stellar evolution specific
VAE_CONFIG_STELLAR = VAEConfig(
    input_dim=6,  # mass, luminosity, temperature, radius, metallicity, age
    hidden_dims=[64, 32, 16],
    latent_dim=4,
    activation='elu',
    batch_size=64,
    num_epochs=150
)

PINN_CONFIG_STELLAR = PINNConfig(
    input_dim=6,
    hidden_dims=[64, 64, 32],
    activation='tanh',
    physics_weight=0.5,  # Strong physics constraints
    batch_size=64,
    num_epochs=150
)

# Exoplanet transit specific
VAE_CONFIG_EXOPLANET = VAEConfig(
    input_dim=6,  # period, depth, duration, impact_parameter, planet_radius, stellar_radius
    hidden_dims=[48, 24, 12],
    latent_dim=4,
    batch_size=32,
    num_epochs=100
)

PINN_CONFIG_EXOPLANET = PINNConfig(
    input_dim=6,
    hidden_dims=[48, 48, 24],
    physics_weight=0.3,
    batch_size=32,
    num_epochs=100
)

# Configuration registry
VAE_PRESETS = {
    'small': VAE_CONFIG_SMALL,
    'medium': VAE_CONFIG_MEDIUM,
    'large': VAE_CONFIG_LARGE,
    'stellar': VAE_CONFIG_STELLAR,
    'exoplanet': VAE_CONFIG_EXOPLANET
}

PINN_PRESETS = {
    'small': PINN_CONFIG_SMALL,
    'medium': PINN_CONFIG_MEDIUM,
    'large': PINN_CONFIG_LARGE,
    'stellar': PINN_CONFIG_STELLAR,
    'exoplanet': PINN_CONFIG_EXOPLANET
}

def get_vae_config(preset_name: str = 'medium') -> VAEConfig:
    """Get a preset VAE configuration"""
    if preset_name not in VAE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(VAE_PRESETS.keys())}")
    return VAE_PRESETS[preset_name]

def get_pinn_config(preset_name: str = 'medium') -> PINNConfig:
    """Get a preset PINN configuration"""
    if preset_name not in PINN_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PINN_PRESETS.keys())}")
    return PINN_PRESETS[preset_name]
