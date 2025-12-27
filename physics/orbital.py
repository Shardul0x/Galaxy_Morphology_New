# physics/orbital.py
"""
Orbital mechanics physics constraints
Includes Kepler's laws, conservation laws, and orbital dynamics
"""

import torch
import torch.nn as nn
from .base import PhysicsEquation, ConservationLaw
import numpy as np


class KeplersThirdLaw(PhysicsEquation):
    """
    Kepler's Third Law: P² = (4π²/GM) * a³
    
    For solar system units (M = M☉):
    P² (years) = a³ (AU)
    """
    
    def __init__(
        self,
        period_index: int = 0,
        semi_major_axis_index: int = 1,
        stellar_mass: float = 1.0
    ):
        super().__init__(
            name="keplers_third_law",
            description="Kepler's third law for orbital mechanics"
        )
        self.period_index = period_index
        self.semi_major_axis_index = semi_major_axis_index
        self.stellar_mass = stellar_mass
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Kepler's third law residual
        
        P² = a³ / M (in solar system units)
        """
        period = features[:, self.period_index]
        semi_major_axis = features[:, self.semi_major_axis_index]
        
        # Expected period squared from Kepler's law
        expected_P_squared = (semi_major_axis ** 3) / self.stellar_mass
        observed_P_squared = period ** 2
        
        # Relative residual (more stable for wide range of values)
        residual = (observed_P_squared - expected_P_squared) / (expected_P_squared + 1e-6)
        
        return residual


class OrbitalEnergyConservation(ConservationLaw):
    """
    Orbital energy conservation: E = -GM/(2a) = constant
    
    For circular orbits: E = (v²/2) - (GM/r)
    """
    
    def __init__(
        self,
        semi_major_axis_index: int = 0,
        velocity_index: int = 2,
        stellar_mass: float = 1.0
    ):
        def energy_fn(features):
            a = features[:, semi_major_axis_index]
            # Orbital energy: E = -GM/(2a)
            # In solar units: E = -M/(2a)
            return -stellar_mass / (2 * a + 1e-6)
        
        super().__init__(
            name="orbital_energy_conservation",
            conserved_quantity_fn=energy_fn,
            description="Conservation of orbital energy"
        )
        self.stellar_mass = stellar_mass


class AngularMomentumConservation(ConservationLaw):
    """
    Angular momentum conservation: L = r × mv = constant
    
    For Keplerian orbits: L = √(GMa(1-e²))
    """
    
    def __init__(
        self,
        semi_major_axis_index: int = 0,
        eccentricity_index: int = 1,
        stellar_mass: float = 1.0
    ):
        def angular_momentum_fn(features):
            a = features[:, semi_major_axis_index]
            e = features[:, eccentricity_index]
            # L = √(GMa(1-e²))
            # In solar units: L = √(M*a(1-e²))
            return torch.sqrt(stellar_mass * a * (1 - e**2) + 1e-6)
        
        super().__init__(
            name="angular_momentum_conservation",
            conserved_quantity_fn=angular_momentum_fn,
            description="Conservation of orbital angular momentum"
        )


class VisVivaEquation(PhysicsEquation):
    """
    Vis-viva equation: v² = GM(2/r - 1/a)
    
    Relates orbital velocity to position and semi-major axis
    """
    
    def __init__(
        self,
        velocity_index: int = 2,
        distance_index: int = 3,
        semi_major_axis_index: int = 0,
        stellar_mass: float = 1.0
    ):
        super().__init__(
            name="vis_viva_equation",
            description="Vis-viva equation for orbital velocity"
        )
        self.velocity_index = velocity_index
        self.distance_index = distance_index
        self.semi_major_axis_index = semi_major_axis_index
        self.stellar_mass = stellar_mass
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute vis-viva equation residual
        
        v² = GM(2/r - 1/a)
        """
        velocity = features[:, self.velocity_index]
        distance = features[:, self.distance_index]
        semi_major_axis = features[:, self.semi_major_axis_index]
        
        # Expected v² from vis-viva
        expected_v_squared = self.stellar_mass * (2.0 / (distance + 1e-6) - 1.0 / (semi_major_axis + 1e-6))
        observed_v_squared = velocity ** 2
        
        residual = observed_v_squared - expected_v_squared
        
        return residual


class EccentricityBounds(PhysicsEquation):
    """
    Eccentricity must be in range [0, 1) for bound orbits
    """
    
    def __init__(self, eccentricity_index: int = 1):
        super().__init__(
            name="eccentricity_bounds",
            description="Eccentricity bounds for elliptical orbits"
        )
        self.eccentricity_index = eccentricity_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Penalize eccentricity outside [0, 1)
        """
        e = features[:, self.eccentricity_index]
        
        # Penalty for e < 0
        below_penalty = torch.relu(-e)
        
        # Penalty for e >= 1 (hyperbolic/parabolic)
        above_penalty = torch.relu(e - 0.99)
        
        return below_penalty + above_penalty


class TransitGeometry(PhysicsEquation):
    """
    Transit geometry constraints for exoplanets
    
    Transit depth: δ = (R_p / R_star)²
    Transit duration: T ∝ P/π * √(1 - b²)
    """
    
    def __init__(
        self,
        depth_index: int = 1,
        planet_radius_index: int = 4,
        stellar_radius_index: int = 5
    ):
        super().__init__(
            name="transit_geometry",
            description="Transit depth geometry"
        )
        self.depth_index = depth_index
        self.planet_radius_index = planet_radius_index
        self.stellar_radius_index = stellar_radius_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Check transit depth consistency: δ = (R_p / R_star)²
        """
        depth = features[:, self.depth_index]
        R_p = features[:, self.planet_radius_index]
        R_star = features[:, self.stellar_radius_index]
        
        # Expected depth (convert R_earth to R_sun: 1 R_earth ≈ 0.00916 R_sun)
        R_p_solar = R_p * 0.00916
        expected_depth = (R_p_solar / R_star) ** 2
        
        residual = depth - expected_depth
        
        return residual


# Factory functions

def keplers_third_law(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Kepler's third law"""
    constraint = KeplersThirdLaw()
    return constraint.loss(features)


def orbital_energy_conservation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for energy conservation"""
    constraint = OrbitalEnergyConservation()
    return constraint.loss(features, time_derivative=time_derivative)


def angular_momentum_conservation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for angular momentum conservation"""
    constraint = AngularMomentumConservation()
    return constraint.loss(features, time_derivative=time_derivative)


def vis_viva_equation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for vis-viva equation"""
    constraint = VisVivaEquation()
    return constraint.loss(features)
