# physics/__init__.py
"""
Physics module for astrophysics simulations
Contains physics-based constraints and equations
"""

from .base import PhysicsConstraint, PhysicsEquation
from .stellar import (
    mass_luminosity_relation,
    stefan_boltzmann_law,
    main_sequence_lifetime,
    hertzsprung_russell_constraint
)
from .orbital import (
    keplers_third_law,
    orbital_energy_conservation,
    angular_momentum_conservation,
    vis_viva_equation
)
from .cosmological import (
    hubble_law,
    distance_modulus,
    lookback_time_constraint,
    friedmann_equation
)
from .galaxy import (
    galaxy_rotation_constraint,
    tully_fisher_relation,
    faber_jackson_relation,
    morphology_consistency
)

__all__ = [
    'PhysicsConstraint',
    'PhysicsEquation',
    # Stellar
    'mass_luminosity_relation',
    'stefan_boltzmann_law',
    'main_sequence_lifetime',
    'hertzsprung_russell_constraint',
    # Orbital
    'keplers_third_law',
    'orbital_energy_conservation',
    'angular_momentum_conservation',
    'vis_viva_equation',
    # Cosmological
    'hubble_law',
    'distance_modulus',
    'lookback_time_constraint',
    'friedmann_equation',
    # Galaxy
    'galaxy_rotation_constraint',
    'tully_fisher_relation',
    'faber_jackson_relation',
    'morphology_consistency'
]
