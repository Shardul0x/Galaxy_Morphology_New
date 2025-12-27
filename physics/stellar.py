# physics/stellar.py
"""
Stellar physics constraints and equations
Includes main sequence relations, stellar evolution, and HR diagram constraints
"""

import torch
import torch.nn as nn
from .base import PhysicsEquation, PowerLaw
import numpy as np


class MassLuminosityRelation(PowerLaw):
    """
    Mass-Luminosity Relation: L ∝ M^α
    
    For main sequence stars:
    - Low mass (M < 0.43 M☉): α ≈ 2.3
    - Mid mass (0.43 < M < 2 M☉): α ≈ 4.0
    - High mass (M > 2 M☉): α ≈ 3.5
    """
    
    def __init__(self, mass_index: int = 0, luminosity_index: int = 1, alpha: float = 3.5):
        super().__init__(
            name="mass_luminosity_relation",
            x_index=mass_index,
            y_index=luminosity_index,
            alpha=alpha,
            k=1.0,
            description="Main sequence mass-luminosity relation"
        )
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute mass-luminosity residual with mass-dependent exponent
        """
        mass = features[:, self.x_index]
        luminosity = features[:, self.y_index]
        
        # Mass-dependent exponent
        alpha = torch.where(mass < 0.43, 2.3, 
                           torch.where(mass < 2.0, 4.0, 3.5))
        
        expected_luminosity = mass ** alpha
        residual = luminosity - expected_luminosity
        
        return residual


class StefanBoltzmannLaw(PhysicsEquation):
    """
    Stefan-Boltzmann Law: L = 4πR²σT⁴
    In solar units: L ∝ R² T⁴
    """
    
    def __init__(
        self,
        luminosity_index: int = 1,
        temperature_index: int = 2,
        radius_index: int = 3,
        T_sun: float = 5778.0
    ):
        super().__init__(
            name="stefan_boltzmann_law",
            description="Stefan-Boltzmann law for stellar luminosity"
        )
        self.luminosity_index = luminosity_index
        self.temperature_index = temperature_index
        self.radius_index = radius_index
        self.T_sun = T_sun
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Stefan-Boltzmann residual
        
        L = 4πR²σT⁴
        In solar units: L/L☉ = (R/R☉)² * (T/T☉)⁴
        """
        luminosity = features[:, self.luminosity_index]
        temperature = features[:, self.temperature_index]
        radius = features[:, self.radius_index]
        
        # Normalize temperature to solar temperature
        T_norm = temperature / self.T_sun
        
        # Expected luminosity from Stefan-Boltzmann
        expected_luminosity = (radius ** 2) * (T_norm ** 4)
        
        residual = luminosity - expected_luminosity
        
        return residual


class MainSequenceLifetime(PhysicsEquation):
    """
    Main sequence lifetime: τ ∝ M / L ≈ M^(-2.5)
    
    More massive stars burn faster and have shorter lifetimes
    """
    
    def __init__(self, mass_index: int = 0, age_index: int = 5, max_age: float = 13.8):
        super().__init__(
            name="main_sequence_lifetime",
            description="Main sequence stellar lifetime constraint"
        )
        self.mass_index = mass_index
        self.age_index = age_index
        self.max_age = max_age  # Age of universe in Gyr
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute lifetime constraint
        
        τ_MS ≈ 10 Gyr * (M/M☉)^(-2.5)
        Age should be less than min(τ_MS, age_of_universe)
        """
        mass = features[:, self.mass_index]
        age = features[:, self.age_index]
        
        # Main sequence lifetime in Gyr
        lifetime = 10.0 * (mass ** (-2.5))
        
        # Maximum possible age
        max_possible_age = torch.minimum(lifetime, torch.tensor(self.max_age))
        
        # Penalty for age exceeding maximum
        violation = torch.relu(age - max_possible_age)
        
        return violation


class HertzsprungRussellConstraint(PhysicsEquation):
    """
    Hertzsprung-Russell diagram constraints
    Ensures stars fall within valid regions of the HR diagram
    """
    
    def __init__(
        self,
        luminosity_index: int = 1,
        temperature_index: int = 2,
        region: str = "main_sequence"
    ):
        super().__init__(
            name="hr_diagram_constraint",
            description=f"HR diagram constraint for {region}"
        )
        self.luminosity_index = luminosity_index
        self.temperature_index = temperature_index
        self.region = region
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute HR diagram constraint
        
        Main sequence approximately follows: log(L) ≈ 4*log(T) - constant
        """
        luminosity = features[:, self.luminosity_index]
        temperature = features[:, self.temperature_index]
        
        # Log values (add small epsilon to avoid log(0))
        log_L = torch.log(luminosity + 1e-6)
        log_T = torch.log(temperature + 1e-6)
        
        if self.region == "main_sequence":
            # Main sequence relation: log(L) ≈ 4*log(T) - constant
            # For T_sun = 5778K, L_sun = 1: constant ≈ 4*log(5778) ≈ 34.6
            expected_log_L = 4.0 * log_T - 34.6
            
            # Allow some deviation (stars can be off main sequence)
            residual = torch.abs(log_L - expected_log_L)
            
            # Soft constraint (only penalize large deviations)
            penalty = torch.relu(residual - 2.0)  # Allow ±2 in log space
            
        elif self.region == "giant_branch":
            # Giants: high luminosity, low temperature
            # L > 100, T < 5000
            lum_penalty = torch.relu(100.0 - luminosity)
            temp_penalty = torch.relu(temperature - 5000.0)
            penalty = lum_penalty + temp_penalty
            
        else:
            penalty = torch.zeros_like(luminosity)
        
        return penalty


class MetallicityEvolution(PhysicsEquation):
    """
    Metallicity evolution constraint
    Older stars tend to have lower metallicity (Population II)
    """
    
    def __init__(self, metallicity_index: int = 4, age_index: int = 5):
        super().__init__(
            name="metallicity_evolution",
            description="Age-metallicity relation"
        )
        self.metallicity_index = metallicity_index
        self.age_index = age_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Older stars should have lower metallicity
        
        Approximate relation: [Fe/H] ≈ -0.2 * age (Gyr) + 0.2
        """
        metallicity = features[:, self.metallicity_index]
        age = features[:, self.age_index]
        
        # Expected metallicity based on age
        expected_metallicity = -0.2 * age + 0.2
        
        # Soft constraint (large scatter is physical)
        residual = metallicity - expected_metallicity
        
        # Only penalize extreme violations
        penalty = torch.relu(torch.abs(residual) - 1.0)
        
        return penalty


# Factory functions for easy use

def mass_luminosity_relation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for mass-luminosity constraint"""
    constraint = MassLuminosityRelation()
    return constraint.loss(features)


def stefan_boltzmann_law(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Stefan-Boltzmann constraint"""
    constraint = StefanBoltzmannLaw()
    return constraint.loss(features)


def main_sequence_lifetime(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for main sequence lifetime constraint"""
    constraint = MainSequenceLifetime()
    return constraint.loss(features)


def hertzsprung_russell_constraint(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for HR diagram constraint"""
    constraint = HertzsprungRussellConstraint()
    return constraint.loss(features)
