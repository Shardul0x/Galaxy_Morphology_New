# models/losses.py
"""
Loss Functions for Astrophysics Data Augmentation
Includes VAE losses, PINN losses, and physics-based constraints
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import numpy as np

# ===== VAE LOSSES =====

def vae_loss_standard(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard VAE loss: Reconstruction + β * KL Divergence
    
    Args:
        reconstruction: Reconstructed features [batch_size, feature_dim]
        original: Original features [batch_size, feature_dim]
        mu: Mean of latent distribution [batch_size, latent_dim]
        logvar: Log variance of latent distribution [batch_size, latent_dim]
        beta: Weight for KL divergence (β-VAE parameter)
    
    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
    
    # KL divergence: KL(N(μ,σ²) || N(0,1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / original.size(0)  # Average over batch
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def vae_loss_weighted_features(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    feature_weights: Optional[torch.Tensor] = None,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss with weighted feature reconstruction
    Useful when some features are more important than others
    
    Args:
        reconstruction: Reconstructed features
        original: Original features
        mu: Latent mean
        logvar: Latent log variance
        feature_weights: Weights for each feature [feature_dim]
        beta: KL divergence weight
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Weighted reconstruction loss
    if feature_weights is None:
        feature_weights = torch.ones(original.size(1), device=original.device)
    
    # Expand weights to match batch size
    weights = feature_weights.unsqueeze(0).expand_as(original)
    
    # Weighted MSE
    recon_loss = torch.mean(weights * (reconstruction - original) ** 2)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / original.size(0)
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def vae_loss_disentangled(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
    capacity: float = 0.0,
    gamma: float = 1000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE loss with controlled capacity for disentanglement
    Based on "Understanding disentangling in β-VAE" (Burgess et al., 2018)
    
    Args:
        reconstruction: Reconstructed features
        original: Original features
        mu: Latent mean
        logvar: Latent log variance
        beta: Disentanglement weight
        capacity: Target capacity (gradually increased during training)
        gamma: Regularization weight
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
    
    # KL divergence with capacity constraint
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence = kl_divergence / original.size(0)
    
    # Capacity-constrained KL loss
    kl_loss = gamma * torch.abs(kl_divergence - capacity)
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ===== PINN LOSSES =====

def pinn_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Data fitting loss for PINN
    
    Args:
        prediction: Predicted features
        target: Target features
        loss_type: 'mse', 'mae', or 'huber'
    
    Returns:
        Data loss
    """
    if loss_type == 'mse':
        return nn.functional.mse_loss(prediction, target)
    elif loss_type == 'mae':
        return nn.functional.l1_loss(prediction, target)
    elif loss_type == 'huber':
        return nn.functional.smooth_l1_loss(prediction, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def pinn_temporal_smoothness_loss(
    evolved_states: torch.Tensor
) -> torch.Tensor:
    """
    Enforce temporal smoothness in evolution
    Penalizes large jumps between consecutive time steps
    
    Args:
        evolved_states: Time series of evolved states [num_steps, batch_size, feature_dim]
    
    Returns:
        Smoothness loss
    """
    # Compute differences between consecutive time steps
    time_diffs = evolved_states[1:] - evolved_states[:-1]
    
    # L2 norm of differences
    smoothness_loss = torch.mean(time_diffs ** 2)
    
    return smoothness_loss


def pinn_conservation_loss(
    evolved_states: torch.Tensor,
    conserved_quantity_fn: Callable
) -> torch.Tensor:
    """
    Enforce conservation of a physical quantity
    
    Args:
        evolved_states: Time series [num_steps, batch_size, feature_dim]
        conserved_quantity_fn: Function that computes the conserved quantity
    
    Returns:
        Conservation loss
    """
    # Compute conserved quantity at each time step
    quantities = torch.stack([
        conserved_quantity_fn(state) for state in evolved_states
    ])
    
    # Variance of conserved quantity (should be zero)
    conservation_loss = torch.var(quantities)
    
    return conservation_loss


# ===== PHYSICS-BASED LOSSES =====

def mass_luminosity_loss(
    features: torch.Tensor,
    mass_idx: int = 0,
    luminosity_idx: int = 1,
    alpha: float = 3.5
) -> torch.Tensor:
    """
    Mass-Luminosity relation: L ∝ M^α
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        mass_idx: Index of mass feature
        luminosity_idx: Index of luminosity feature
        alpha: Power law exponent (3.5 for main sequence)
    
    Returns:
        Physics loss
    """
    mass = features[:, mass_idx]
    luminosity = features[:, luminosity_idx]
    
    # Expected luminosity from mass-luminosity relation
    expected_luminosity = mass ** alpha
    
    # MSE between observed and expected
    loss = nn.functional.mse_loss(luminosity, expected_luminosity)
    
    return loss


def kepler_third_law_loss(
    features: torch.Tensor,
    semi_major_axis_idx: int = 0,
    period_idx: int = 1,
    stellar_mass: float = 1.0  # In solar masses
) -> torch.Tensor:
    """
    Kepler's Third Law: P² = (4π²/GM) * a³
    For a/M in solar units: P² ∝ a³/M
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        semi_major_axis_idx: Index of semi-major axis (AU)
        period_idx: Index of period (years)
        stellar_mass: Stellar mass in solar masses
    
    Returns:
        Physics loss
    """
    a = features[:, semi_major_axis_idx]
    P = features[:, period_idx]
    
    # P² should equal a³/M (in solar system units)
    expected_period_squared = (a ** 3) / stellar_mass
    observed_period_squared = P ** 2
    
    loss = nn.functional.mse_loss(observed_period_squared, expected_period_squared)
    
    return loss


def stefan_boltzmann_loss(
    features: torch.Tensor,
    luminosity_idx: int = 1,
    temperature_idx: int = 2,
    radius_idx: int = 3
) -> torch.Tensor:
    """
    Stefan-Boltzmann Law: L = 4πR²σT⁴
    In solar units: L ∝ R² T⁴
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        luminosity_idx: Index of luminosity
        temperature_idx: Index of temperature
        radius_idx: Index of radius
    
    Returns:
        Physics loss
    """
    luminosity = features[:, luminosity_idx]
    temperature = features[:, temperature_idx]
    radius = features[:, radius_idx]
    
    # Expected luminosity from Stefan-Boltzmann
    # L ∝ R² T⁴ (in solar units, with T in units of T_sun)
    T_sun = 5778.0  # Kelvin
    temperature_normalized = temperature / T_sun
    expected_luminosity = (radius ** 2) * (temperature_normalized ** 4)
    
    loss = nn.functional.mse_loss(luminosity, expected_luminosity)
    
    return loss


def energy_conservation_loss(
    features: torch.Tensor,
    time_derivative: torch.Tensor,
    kinetic_idx: int = 0,
    potential_idx: int = 1
) -> torch.Tensor:
    """
    Energy conservation: d(KE + PE)/dt = 0
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        time_derivative: Time derivatives dx/dt [batch_size, feature_dim]
        kinetic_idx: Index of kinetic energy
        potential_idx: Index of potential energy
    
    Returns:
        Physics loss
    """
    # Rate of change of total energy
    dKE_dt = time_derivative[:, kinetic_idx]
    dPE_dt = time_derivative[:, potential_idx]
    dE_dt = dKE_dt + dPE_dt
    
    # Energy conservation: dE/dt should be zero
    loss = torch.mean(dE_dt ** 2)
    
    return loss


def angular_momentum_conservation_loss(
    features: torch.Tensor,
    time_derivative: torch.Tensor,
    position_idx: tuple = (0, 1, 2),
    velocity_idx: tuple = (3, 4, 5),
    mass_idx: int = 6
) -> torch.Tensor:
    """
    Angular momentum conservation: L = r × mv = constant
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        time_derivative: Time derivatives
        position_idx: Indices for (x, y, z) position
        velocity_idx: Indices for (vx, vy, vz) velocity
        mass_idx: Index of mass
    
    Returns:
        Physics loss
    """
    # Extract position and velocity
    r = features[:, list(position_idx)]
    v = features[:, list(velocity_idx)]
    m = features[:, mass_idx].unsqueeze(1)
    
    # Compute angular momentum L = r × (mv)
    L = torch.cross(r, m * v, dim=1)
    
    # Angular momentum should be constant (variance = 0)
    loss = torch.var(L, dim=0).mean()
    
    return loss


# ===== COMPOSITE LOSS FUNCTIONS =====

class CompositeLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights
    """
    
    def __init__(self, loss_functions: dict):
        """
        Args:
            loss_functions: Dictionary of {name: (function, weight)}
        """
        super().__init__()
        self.loss_functions = loss_functions
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted sum of all losses
        
        Returns:
            total_loss: Weighted sum
            loss_dict: Individual loss values
        """
        total_loss = 0.0
        loss_dict = {}
        
        for name, (loss_fn, weight) in self.loss_functions.items():
            loss_value = loss_fn(*args, **kwargs)
            loss_dict[name] = loss_value.item()
            total_loss += weight * loss_value
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ===== HELPER FUNCTIONS =====

def get_beta_schedule(epoch: int, num_epochs: int, schedule_type: str = 'linear', max_beta: float = 1.0) -> float:
    """
    Get β value for β-VAE based on training schedule
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        schedule_type: 'linear', 'cyclical', or 'constant'
        max_beta: Maximum β value
    
    Returns:
        Current β value
    """
    if schedule_type == 'constant':
        return max_beta
    
    elif schedule_type == 'linear':
        # Linear warmup
        return min(max_beta, max_beta * epoch / (num_epochs * 0.5))
    
    elif schedule_type == 'cyclical':
        # Cyclical schedule (4 cycles)
        cycle_length = num_epochs / 4
        position = (epoch % cycle_length) / cycle_length
        return max_beta * position
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
