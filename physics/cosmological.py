# physics/cosmological.py
"""
Cosmological physics constraints
Includes Hubble law, distance measures, and Friedmann equations
"""

import torch
import torch.nn as nn
from .base import PhysicsEquation
import numpy as np


class HubbleLaw(PhysicsEquation):
    """
    Hubble's Law: v = H₀ * d
    
    Relates recession velocity to distance
    """
    
    def __init__(
        self,
        redshift_index: int = 0,
        distance_index: int = 1,
        H0: float = 70.0  # km/s/Mpc
    ):
        super().__init__(
            name="hubble_law",
            description="Hubble's law relating redshift to distance"
        )
        self.redshift_index = redshift_index
        self.distance_index = distance_index
        self.H0 = H0
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        For low redshift: z ≈ H₀ * d / c
        """
        z = features[:, self.redshift_index]
        d = features[:, self.distance_index]  # Mpc
        
        # Speed of light in km/s
        c = 299792.458
        
        # Expected redshift from Hubble law (low-z approximation)
        expected_z = (self.H0 * d) / c
        
        # Only apply to low redshift (z < 0.1)
        mask = (z < 0.1).float()
        residual = mask * (z - expected_z)
        
        return residual


class DistanceModulus(PhysicsEquation):
    """
    Distance modulus: μ = 5 * log₁₀(d_L) + 25
    
    Relates luminosity distance to apparent magnitude
    """
    
    def __init__(
        self,
        luminosity_distance_index: int = 2,
        distance_modulus_index: int = 6
    ):
        super().__init__(
            name="distance_modulus",
            description="Distance modulus relation"
        )
        self.luminosity_distance_index = luminosity_distance_index
        self.distance_modulus_index = distance_modulus_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        μ = 5 * log₁₀(d_L / 10 pc)
        μ = 5 * log₁₀(d_L [Mpc]) + 25
        """
        d_L = features[:, self.luminosity_distance_index]
        mu = features[:, self.distance_modulus_index]
        
        # Expected distance modulus
        expected_mu = 5.0 * torch.log10(d_L + 1e-6) + 25.0
        
        residual = mu - expected_mu
        
        return residual


class LookbackTimeConstraint(PhysicsEquation):
    """
    Lookback time consistency with redshift and age
    
    Higher redshift → longer lookback time
    Lookback time < age of universe
    """
    
    def __init__(
        self,
        redshift_index: int = 0,
        lookback_time_index: int = 4,
        age_universe: float = 13.8  # Gyr
    ):
        super().__init__(
            name="lookback_time_constraint",
            description="Lookback time consistency"
        )
        self.redshift_index = redshift_index
        self.lookback_time_index = lookback_time_index
        self.age_universe = age_universe
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Lookback time should increase with redshift
        and be less than age of universe
        """
        z = features[:, self.redshift_index]
        t_lookback = features[:, self.lookback_time_index]
        
        # Approximate lookback time (flat universe, matter dominated)
        # t_lookback ≈ (2/3H₀) * [1 - 1/(1+z)^(3/2)]
        # Simplified: t ≈ 13 * z / (1 + z)  for small z
        expected_t = 13.0 * z / (1.0 + z)
        
        # Residual
        residual = t_lookback - expected_t
        
        # Also penalize if exceeds age of universe
        violation = torch.relu(t_lookback - self.age_universe)
        
        return residual + violation


class FriedmannEquation(PhysicsEquation):
    """
    Friedmann equation (simplified for flat universe)
    
    H²(z) = H₀² * [Ω_m * (1+z)³ + Ω_Λ]
    """
    
    def __init__(
        self,
        redshift_index: int = 0,
        H0: float = 70.0,
        Omega_m: float = 0.3,
        Omega_Lambda: float = 0.7
    ):
        super().__init__(
            name="friedmann_equation",
            description="Friedmann equation for expanding universe"
        )
        self.redshift_index = redshift_index
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Check Friedmann equation consistency
        
        This is more of a self-consistency check for derived quantities
        """
        z = features[:, self.redshift_index]
        
        # Hubble parameter at redshift z
        H_z = self.H0 * torch.sqrt(
            self.Omega_m * (1 + z)**3 + self.Omega_Lambda
        )
        
        # Return relative deviation (placeholder for full implementation)
        # In practice, this would compare with other distance measures
        return torch.zeros_like(z)


class AngularDiameterDistance(PhysicsEquation):
    """
    Angular diameter distance relation
    
    d_A = d_L / (1 + z)²
    """
    
    def __init__(
        self,
        redshift_index: int = 0,
        angular_diameter_distance_index: int = 1,
        luminosity_distance_index: int = 2
    ):
        super().__init__(
            name="angular_diameter_distance",
            description="Relation between angular and luminosity distances"
        )
        self.redshift_index = redshift_index
        self.angular_diameter_distance_index = angular_diameter_distance_index
        self.luminosity_distance_index = luminosity_distance_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        d_A = d_L / (1 + z)²
        """
        z = features[:, self.redshift_index]
        d_A = features[:, self.angular_diameter_distance_index]
        d_L = features[:, self.luminosity_distance_index]
        
        # Expected angular diameter distance
        expected_d_A = d_L / ((1.0 + z) ** 2)
        
        residual = d_A - expected_d_A
        
        return residual


# Factory functions

def hubble_law(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Hubble's law"""
    constraint = HubbleLaw()
    return constraint.loss(features)


def distance_modulus(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for distance modulus"""
    constraint = DistanceModulus()
    return constraint.loss(features)


def lookback_time_constraint(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for lookback time"""
    constraint = LookbackTimeConstraint()
    return constraint.loss(features)


def friedmann_equation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Friedmann equation"""
    constraint = FriedmannEquation()
    return constraint.loss(features)
