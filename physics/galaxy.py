# physics/galaxy.py
"""
Galaxy physics constraints
Includes scaling relations, rotation curves, and morphology constraints
"""

import torch
import torch.nn as nn
from .base import PhysicsEquation, PowerLaw
import numpy as np


class GalaxyRotationConstraint(PhysicsEquation):
    """
    Flat rotation curve constraint
    
    For spiral galaxies, rotation velocity is approximately constant
    in the outer regions (dark matter halo)
    """
    
    def __init__(self, rotation_velocity_index: int = 0):
        super().__init__(
            name="galaxy_rotation_constraint",
            description="Flat rotation curve for spiral galaxies"
        )
        self.rotation_velocity_index = rotation_velocity_index
    
    def compute(self, features: torch.Tensor, time_derivative: torch.Tensor = None) -> torch.Tensor:
        """
        Check that rotation velocity doesn't change too rapidly with time
        (galaxies evolve slowly)
        """
        if time_derivative is not None:
            dv_dt = time_derivative[:, self.rotation_velocity_index]
            # Rotation velocity should be stable
            return torch.abs(dv_dt)
        else:
            return torch.zeros(features.size(0), device=features.device)


class TullyFisherRelation(PowerLaw):
    """
    Tully-Fisher Relation: L ∝ v^α
    
    For spiral galaxies: L ∝ v⁴ (α ≈ 4)
    Relates luminosity to rotation velocity
    """
    
    def __init__(
        self,
        luminosity_index: int = 1,
        rotation_velocity_index: int = 0,
        alpha: float = 4.0
    ):
        super().__init__(
            name="tully_fisher_relation",
            x_index=rotation_velocity_index,
            y_index=luminosity_index,
            alpha=alpha,
            k=1.0,
            description="Tully-Fisher relation for spiral galaxies"
        )


class FaberJacksonRelation(PowerLaw):
    """
    Faber-Jackson Relation: L ∝ σ^α
    
    For elliptical galaxies: L ∝ σ⁴ (α ≈ 4)
    Relates luminosity to velocity dispersion
    """
    
    def __init__(
        self,
        luminosity_index: int = 1,
        velocity_dispersion_index: int = 2,
        alpha: float = 4.0
    ):
        super().__init__(
            name="faber_jackson_relation",
            x_index=velocity_dispersion_index,
            y_index=luminosity_index,
            alpha=alpha,
            k=1.0,
            description="Faber-Jackson relation for elliptical galaxies"
        )


class MorphologyConsistency(PhysicsEquation):
    """
    Galaxy morphology consistency constraints
    
    Ensures morphological features are self-consistent
    E.g., elliptical galaxies shouldn't have strong spiral arms
    """
    
    def __init__(
        self,
        P_EL_index: int = 0,
        P_CW_index: int = 1,
        P_ACW_index: int = 2
    ):
        super().__init__(
            name="morphology_consistency",
            description="Galaxy morphology self-consistency"
        )
        self.P_EL_index = P_EL_index
        self.P_CW_index = P_CW_index
        self.P_ACW_index = P_ACW_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Check morphology consistency:
        - High P_EL (elliptical) → low P_CW and P_ACW (spiral arms)
        - High spiral probability → low elliptical probability
        """
        P_EL = features[:, self.P_EL_index]
        P_CW = features[:, self.P_CW_index]
        P_ACW = features[:, self.P_ACW_index]
        
        # Elliptical galaxies shouldn't have strong spiral arms
        elliptical_spiral_violation = P_EL * (P_CW + P_ACW)
        
        # Spiral galaxies shouldn't be highly elliptical
        spiral_elliptical_violation = (P_CW + P_ACW) * P_EL
        
        # Total violation
        violation = elliptical_spiral_violation + spiral_elliptical_violation
        
        # Soft constraint (only penalize strong violations)
        penalty = torch.relu(violation - 0.5)
        
        return penalty


class BulgeDiscRelation(PhysicsEquation):
    """
    Bulge-to-disc ratio constraints
    
    More elliptical galaxies have larger bulges
    """
    
    def __init__(
        self,
        P_EL_index: int = 0,
        P_CS_index: int = 6  # Central concentration
    ):
        super().__init__(
            name="bulge_disc_relation",
            description="Bulge-disc ratio relation"
        )
        self.P_EL_index = P_EL_index
        self.P_CS_index = P_CS_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Higher ellipticity should correlate with higher central concentration
        """
        P_EL = features[:, self.P_EL_index]
        P_CS = features[:, self.P_CS_index]
        
        # Expected correlation (positive)
        # For ellipticals (P_EL high), expect high P_CS
        expected_correlation = P_EL * 0.5 + 0.2
        
        # Penalize if P_CS is much lower than expected for ellipticals
        violation = torch.relu(expected_correlation - P_CS) * P_EL
        
        return violation


class ColorMagnitudeRelation(PhysicsEquation):
    """
    Color-magnitude relation for galaxies
    
    Red sequence: passive ellipticals
    Blue cloud: star-forming spirals
    """
    
    def __init__(
        self,
        color_index: int = 0,
        magnitude_index: int = 1,
        P_EL_index: int = 2
    ):
        super().__init__(
            name="color_magnitude_relation",
            description="Color-magnitude relation"
        )
        self.color_index = color_index
        self.magnitude_index = magnitude_index
        self.P_EL_index = P_EL_index
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Ellipticals should be redder
        Spirals should be bluer
        """
        color = features[:, self.color_index]
        P_EL = features[:, self.P_EL_index]
        
        # Expected color based on morphology
        # Ellipticals (P_EL high) should be red (high color value)
        # Spirals (P_EL low) should be blue (low color value)
        expected_color = P_EL * 2.0 + 0.5
        
        residual = color - expected_color
        
        return residual


# Factory functions

def galaxy_rotation_constraint(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for rotation constraint"""
    constraint = GalaxyRotationConstraint()
    return constraint.loss(features, time_derivative=time_derivative)


def tully_fisher_relation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Tully-Fisher relation"""
    constraint = TullyFisherRelation()
    return constraint.loss(features)


def faber_jackson_relation(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for Faber-Jackson relation"""
    constraint = FaberJacksonRelation()
    return constraint.loss(features)


def morphology_consistency(features: torch.Tensor, time_derivative: torch.Tensor) -> torch.Tensor:
    """Factory function for morphology consistency"""
    constraint = MorphologyConsistency()
    return constraint.loss(features)
