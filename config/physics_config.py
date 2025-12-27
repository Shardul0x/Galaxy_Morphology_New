# config/physics_config.py
"""
Generic Physics Configuration for Astrophysics Simulations
Supports multiple astrophysics domains with customizable parameters
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np

@dataclass
class PhysicsConstraint:
    """Defines a physics constraint for PINN"""
    name: str
    equation: Callable  # Function that computes the physics loss
    weight: float = 1.0
    description: str = ""

@dataclass
class PhysicsDomain:
    """
    Configures the physics domain for simulation
    
    Examples:
        - Stellar Evolution: mass, luminosity, temperature, radius, age
        - Galaxy Morphology: ellipticity, spiral arm strength, concentration
        - Exoplanet Transits: period, depth, duration, impact parameter
        - Orbital Dynamics: semi-major axis, eccentricity, inclination
    """
    name: str
    feature_names: List[str]
    feature_ranges: Dict[str, tuple]  # Min/max for each feature
    physical_units: Dict[str, str]
    constraints: List[PhysicsConstraint] = field(default_factory=list)
    time_scale: str = "normalized"  # or "years", "days", "Gyr", etc.
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate that features are within physical ranges"""
        for i, name in enumerate(self.feature_names):
            min_val, max_val = self.feature_ranges[name]
            if not (min_val <= features[i] <= max_val):
                return False
        return True

# ===== PREDEFINED PHYSICS DOMAINS =====

def stellar_luminosity_constraint(features, time_derivative):
    """
    Mass-Luminosity Relation: L ∝ M^α
    For main sequence stars: α ≈ 3.5
    """
    mass_idx = 0  # Assuming mass is first feature
    luminosity_idx = 1
    
    mass = features[:, mass_idx]
    luminosity = features[:, luminosity_idx]
    
    # Expected luminosity from mass-luminosity relation
    expected_luminosity = mass ** 3.5
    
    # Physics loss: deviation from expected relation
    loss = ((luminosity - expected_luminosity) ** 2).mean()
    return loss

def orbital_kepler_constraint(features, time_derivative):
    """
    Kepler's Third Law: P² ∝ a³
    For orbital systems
    """
    semi_major_axis_idx = 0
    period_idx = 1
    
    a = features[:, semi_major_axis_idx]
    P = features[:, period_idx]
    
    # P² should be proportional to a³
    expected_ratio = a ** 3
    observed_ratio = P ** 2
    
    loss = ((observed_ratio - expected_ratio) ** 2).mean()
    return loss

def energy_conservation_constraint(features, time_derivative):
    """
    Generic energy conservation constraint
    Ensures total energy remains constant during evolution
    """
    # Assuming features contain kinetic and potential energy terms
    total_energy = features.sum(dim=1)
    
    # Energy should not change significantly over time
    energy_derivative = time_derivative.sum(dim=1)
    loss = (energy_derivative ** 2).mean()
    return loss

# ===== DOMAIN CONFIGURATIONS =====

STELLAR_EVOLUTION = PhysicsDomain(
    name="stellar_evolution",
    feature_names=["mass", "luminosity", "temperature", "radius", "metallicity", "age"],
    feature_ranges={
        "mass": (0.08, 100.0),  # Solar masses
        "luminosity": (0.0001, 1000000.0),  # Solar luminosities
        "temperature": (2000.0, 50000.0),  # Kelvin
        "radius": (0.1, 1000.0),  # Solar radii
        "metallicity": (-2.5, 0.5),  # [Fe/H]
        "age": (0.0, 13.8),  # Gyr
    },
    physical_units={
        "mass": "M_sun",
        "luminosity": "L_sun",
        "temperature": "K",
        "radius": "R_sun",
        "metallicity": "[Fe/H]",
        "age": "Gyr"
    },
    constraints=[
        PhysicsConstraint(
            name="mass_luminosity_relation",
            equation=stellar_luminosity_constraint,
            weight=1.0,
            description="Main sequence mass-luminosity relation"
        )
    ],
    time_scale="Gyr"
)

GALAXY_MORPHOLOGY = PhysicsDomain(
    name="galaxy_morphology",
    feature_names=[
        'P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
        'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED'
    ],
    feature_ranges={
        'P_EL': (0.0, 1.0),
        'P_CW': (0.0, 1.0),
        'P_ACW': (0.0, 1.0),
        'P_EDGE': (0.0, 1.0),
        'P_DK': (0.0, 1.0),
        'P_MG': (0.0, 1.0),
        'P_CS': (0.0, 1.0),
        'P_EL_DEBIASED': (0.0, 1.0),
        'P_CS_DEBIASED': (0.0, 1.0)
    },
    physical_units={
        'P_EL': "probability",
        'P_CW': "probability",
        'P_ACW': "probability",
        'P_EDGE': "probability",
        'P_DK': "probability",
        'P_MG': "probability",
        'P_CS': "probability",
        'P_EL_DEBIASED': "probability",
        'P_CS_DEBIASED': "probability"
    },
    time_scale="Gyr"
)

EXOPLANET_TRANSITS = PhysicsDomain(
    name="exoplanet_transits",
    feature_names=["period", "depth", "duration", "impact_parameter", "planet_radius", "stellar_radius"],
    feature_ranges={
        "period": (0.5, 1000.0),  # days
        "depth": (0.0001, 0.1),  # fractional depth
        "duration": (0.5, 12.0),  # hours
        "impact_parameter": (0.0, 1.0),  # dimensionless
        "planet_radius": (0.5, 20.0),  # Earth radii
        "stellar_radius": (0.1, 10.0),  # Solar radii
    },
    physical_units={
        "period": "days",
        "depth": "fractional",
        "duration": "hours",
        "impact_parameter": "dimensionless",
        "planet_radius": "R_earth",
        "stellar_radius": "R_sun"
    },
    constraints=[
        PhysicsConstraint(
            name="kepler_third_law",
            equation=orbital_kepler_constraint,
            weight=1.0,
            description="Kepler's third law for orbital period"
        )
    ],
    time_scale="days"
)

ORBITAL_DYNAMICS = PhysicsDomain(
    name="orbital_dynamics",
    feature_names=["semi_major_axis", "eccentricity", "inclination", "longitude_ascending", "argument_periapsis", "mean_anomaly"],
    feature_ranges={
        "semi_major_axis": (0.1, 100.0),  # AU
        "eccentricity": (0.0, 0.99),
        "inclination": (0.0, 180.0),  # degrees
        "longitude_ascending": (0.0, 360.0),  # degrees
        "argument_periapsis": (0.0, 360.0),  # degrees
        "mean_anomaly": (0.0, 360.0),  # degrees
    },
    physical_units={
        "semi_major_axis": "AU",
        "eccentricity": "dimensionless",
        "inclination": "degrees",
        "longitude_ascending": "degrees",
        "argument_periapsis": "degrees",
        "mean_anomaly": "degrees"
    },
    time_scale="years"
)

COSMOLOGICAL_PARAMETERS = PhysicsDomain(
    name="cosmological_parameters",
    feature_names=["redshift", "angular_diameter_distance", "luminosity_distance", "age_universe", "lookback_time", "comoving_distance"],
    feature_ranges={
        "redshift": (0.0, 10.0),
        "angular_diameter_distance": (0.0, 30000.0),  # Mpc
        "luminosity_distance": (0.0, 100000.0),  # Mpc
        "age_universe": (0.0, 13.8),  # Gyr
        "lookback_time": (0.0, 13.8),  # Gyr
        "comoving_distance": (0.0, 50000.0),  # Mpc
    },
    physical_units={
        "redshift": "dimensionless",
        "angular_diameter_distance": "Mpc",
        "luminosity_distance": "Mpc",
        "age_universe": "Gyr",
        "lookback_time": "Gyr",
        "comoving_distance": "Mpc"
    },
    time_scale="Gyr"
)

# Registry of available domains
AVAILABLE_DOMAINS = {
    "stellar_evolution": STELLAR_EVOLUTION,
    "galaxy_morphology": GALAXY_MORPHOLOGY,
    "exoplanet_transits": EXOPLANET_TRANSITS,
    "orbital_dynamics": ORBITAL_DYNAMICS,
    "cosmological_parameters": COSMOLOGICAL_PARAMETERS
}

def get_physics_domain(domain_name: str) -> PhysicsDomain:
    """Get a predefined physics domain configuration"""
    if domain_name not in AVAILABLE_DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Available: {list(AVAILABLE_DOMAINS.keys())}")
    return AVAILABLE_DOMAINS[domain_name]
