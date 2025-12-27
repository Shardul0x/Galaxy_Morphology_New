# examples/exoplanet_transits_example.py
"""
Example: Exoplanet Transit Parameter Augmentation
Demonstrates Kepler's law constraints and transit physics
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from config.physics_config import get_physics_domain
from data.augmentation import AstrophysicsAugmentor
from models.vae import GenericVAE
from models.pinn import GenericPINN


def generate_synthetic_exoplanet_data(n_samples=500):
    """
    Generate synthetic exoplanet transit parameters
    In practice, load from NASA Exoplanet Archive or TESS
    """
    np.random.seed(42)
    
    # Period (days) - log-uniform distribution
    period = np.exp(np.random.uniform(np.log(0.5), np.log(1000), n_samples))
    
    # Semi-major axis from Kepler's 3rd law (assuming M_star = 1 M_sun)
    semi_major_axis = (period / 365.25) ** (2/3)  # AU
    
    # Planet radius (Earth radii)
    planet_radius = np.random.lognormal(np.log(2), 0.8, n_samples)
    
    # Stellar radius (Solar radii)
    stellar_radius = np.random.normal(1.0, 0.2, n_samples)
    stellar_radius = np.clip(stellar_radius, 0.5, 2.0)
    
    # Transit depth: (R_p / R_star)^2
    depth = (planet_radius * 0.00916) ** 2 / stellar_radius ** 2  # Convert R_earth to R_sun
    
    # Impact parameter
    impact_parameter = np.random.uniform(0, 0.9, n_samples)
    
    # Transit duration (hours) - simplified formula
    duration = 13 * (period / 365.25) ** (1/3) * np.random.normal(1, 0.2, n_samples)
    duration = np.clip(duration, 0.5, 12)
    
    data = np.column_stack([period, depth, duration, impact_parameter, planet_radius, stellar_radius])
    
    return data


def main():
    print("="*60)
    print("Exoplanet Transit Parameters Augmentation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    print("\n1. Generating synthetic exoplanet transit data...")
    exoplanet_data = generate_synthetic_exoplanet_data(n_samples=500)
    print(f"   Generated {len(exoplanet_data)} exoplanet systems")
    
    # Setup physics
    print("\n2. Loading exoplanet physics constraints...")
    physics_domain = get_physics_domain('exoplanet_transits')
    
    print(f"   Features: {physics_domain.feature_names}")
    print(f"   Constraints: {[c.name for c in physics_domain.constraints]}")
    
    # Preprocess
    from data.preprocessor import AstrophysicsPreprocessor
    
    preprocessor = AstrophysicsPreprocessor(
        scaler_type='minmax',
        log_transform_features=[0, 1],  # period, depth
        clip_outliers=True
    )
    
    data_scaled = preprocessor.fit_transform(exoplanet_data)
    
    # Quick train (simplified for example)
    print("\n3. Training augmentation models...")
    print("   (Using simplified training for demonstration)")
    
    from config.model_config import get_vae_config, get_pinn_config
    
    vae_config = get_vae_config('exoplanet')
    pinn_config = get_pinn_config('exoplanet')
    
    vae = GenericVAE(
        input_dim=vae_config.input_dim,
        hidden_dims=vae_config.hidden_dims,
        latent_dim=vae_config.latent_dim
    )
    
    pinn = GenericPINN(
        input_dim=pinn_config.input_dim,
        hidden_dims=pinn_config.hidden_dims,
        physics_domain=physics_domain
    )
    
    # Simple training (in practice, use full training loop)
    vae.to(device)
    pinn.to(device)
    
    # Augmentation
    print("\n4. Generating augmented exoplanet catalog...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    # Generate new samples
    synthetic_samples = augmentor.generate_samples(
        num_samples=1000,
        validate_physics=True
    )
    
    print(f"   Generated {len(synthetic_samples)} new exoplanet systems")
    
    # Validate physics
    synthetic_original = preprocessor.inverse_transform(synthetic_samples)
    
    # Check Kepler's law
    periods = synthetic_original[:, 0]
    semi_major = (periods / 365.25) ** (2/3)
    
    print("\n5. Physics validation:")
    print(f"   Period range: {periods.min():.2f} - {periods.max():.2f} days")
    print(f"   Implied semi-major axis: {semi_major.min():.3f} - {semi_major.max():.3f} AU")
    print(f"   Transit depth range: {synthetic_original[:, 1].min():.6f} - {synthetic_original[:, 1].max():.4f}")
    
    # Export
    from visualization.export import AstrophysicsExporter
    
    exporter = AstrophysicsExporter(metadata={'domain': 'exoplanet_transits'})
    
    exporter.export_to_csv(
        synthetic_original,
        'results/exoplanet_augmented_catalog.csv',
        column_names=physics_domain.feature_names
    )
    
    print("\n" + "="*60)
    print("Exoplanet Transit Augmentation Complete!")
    print("="*60)


if __name__ == '__main__':
    import os
    os.makedirs('../results', exist_ok=True)
    main()
