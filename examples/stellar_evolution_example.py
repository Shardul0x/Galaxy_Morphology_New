# examples/stellar_evolution_example.py
"""
Example: Stellar Evolution Simulation using VAE + PINN
Demonstrates physics-informed constraints for stellar parameters
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from models.vae import GenericVAE
from models.pinn import GenericPINN
from config.model_config import get_vae_config, get_pinn_config
from config.physics_config import get_physics_domain
from data.loader import AstrophysicsDataLoader
from data.preprocessor import AstrophysicsPreprocessor
from data.augmentation import AstrophysicsAugmentor
from visualization.plotter import AstrophysicsVisualizer


def generate_synthetic_stellar_data(n_samples=1000):
    """
    Generate synthetic stellar parameters for demonstration
    In practice, load from real stellar catalogs (GAIA, APOGEE, etc.)
    """
    np.random.seed(42)
    
    # Generate main sequence stars with physical relations
    mass = np.random.uniform(0.1, 10.0, n_samples)  # Solar masses
    
    # Mass-luminosity relation: L ∝ M^3.5
    luminosity = mass ** 3.5 * np.random.lognormal(0, 0.2, n_samples)
    
    # Mass-temperature relation
    temperature = 5778 * (mass ** 0.5) * np.random.normal(1, 0.1, n_samples)
    
    # Mass-radius relation: R ∝ M^0.8
    radius = mass ** 0.8 * np.random.lognormal(0, 0.15, n_samples)
    
    # Metallicity (solar neighborhood)
    metallicity = np.random.normal(0, 0.3, n_samples)
    
    # Age (younger for more massive stars)
    age = 10.0 / (mass + 0.1) * np.random.uniform(0.5, 1.5, n_samples)
    
    data = np.column_stack([mass, luminosity, temperature, radius, metallicity, age])
    
    return data


def main():
    print("="*60)
    print("Stellar Evolution Example")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Generate/Load Stellar Data
    print("\n1. Generating synthetic stellar data...")
    stellar_data = generate_synthetic_stellar_data(n_samples=1000)
    print(f"   Generated {len(stellar_data)} stellar parameter sets")
    
    # 2. Setup Physics Domain
    print("\n2. Setting up stellar evolution physics...")
    physics_domain = get_physics_domain('stellar_evolution')
    
    print(f"   Features: {physics_domain.feature_names}")
    print(f"   Physics constraints: {len(physics_domain.constraints)}")
    for constraint in physics_domain.constraints:
        print(f"      - {constraint.name}: weight={constraint.weight}")
    
    # 3. Preprocess
    print("\n3. Preprocessing data...")
    
    # Use log transform for mass, luminosity (spanning orders of magnitude)
    preprocessor = AstrophysicsPreprocessor(
        scaler_type='robust',  # Robust to outliers
        log_transform_features=[0, 1, 3],  # mass, luminosity, radius
        clip_outliers=True
    )
    
    stellar_scaled = preprocessor.fit_transform(stellar_data)
    
    # Split data
    split_idx = int(0.8 * len(stellar_scaled))
    X_train = stellar_scaled[:split_idx]
    X_test = stellar_scaled[split_idx:]
    
    # 4. Create Models
    print("\n4. Creating models with stellar-specific configuration...")
    
    vae_config = get_vae_config('stellar')
    pinn_config = get_pinn_config('stellar')
    
    vae = GenericVAE(
        input_dim=vae_config.input_dim,
        hidden_dims=vae_config.hidden_dims,
        latent_dim=vae_config.latent_dim,
        activation=vae_config.activation
    )
    
    pinn = GenericPINN(
        input_dim=pinn_config.input_dim,
        hidden_dims=pinn_config.hidden_dims,
        activation=pinn_config.activation,
        physics_domain=physics_domain
    )
    
    print(f"   VAE latent dim: {vae_config.latent_dim}")
    print(f"   PINN physics weight: {pinn_config.physics_weight}")
    
    # 5. Train Models
    print("\n5. Training models...")
    from torch.utils.data import DataLoader, TensorDataset
    from train import VAETrainer, PINNTrainer
    
    # Train VAE
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=vae_config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test)),
        batch_size=vae_config.batch_size
    )
    
    vae_trainer = VAETrainer(vae, vae_config, device, save_dir='checkpoints/stellar')
    vae_history = vae_trainer.train(train_loader, val_loader)
    
    # Generate PINN training data (stellar evolution sequences)
    print("\n6. Generating stellar evolution sequences for PINN...")
    
    vae.eval()
    n_evolution_samples = 500
    n_timesteps = 30
    
    pinn_data = []
    for i in range(n_evolution_samples):
        idx = np.random.randint(0, len(X_train))
        x = torch.FloatTensor(X_train[idx:idx+1]).to(device)
        
        with torch.no_grad():
            mu, logvar = vae.encode(x)
            
            # Simulate aging by evolving in latent space
            for t in range(n_timesteps):
                time_factor = t / n_timesteps
                
                # Evolve latent representation (simulating stellar aging)
                z = mu + torch.randn_like(mu) * 0.05 * time_factor
                evolved = vae.decode(z)
                
                pinn_data.append((x.cpu(), time_factor, evolved.cpu()))
    
    pinn_X = torch.cat([item[0] for item in pinn_data])
    pinn_t = torch.FloatTensor([[item[1]] for item in pinn_data])
    pinn_y = torch.cat([item[2] for item in pinn_data])
    
    pinn_loader = DataLoader(
        TensorDataset(pinn_X, pinn_t, pinn_y),
        batch_size=pinn_config.batch_size,
        shuffle=True
    )
    
    # Train PINN
    pinn_trainer = PINNTrainer(pinn, pinn_config, physics_domain, device, save_dir='checkpoints/stellar')
    pinn_history = pinn_trainer.train(pinn_loader)
    
    # 7. Data Augmentation
    print("\n7. Augmenting stellar parameter catalog...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    augmented_data, evolutions = augmentor.augment_dataset(
        X_train,
        augmentation_factor=3,
        include_evolution=True,
        evolution_steps=30
    )
    
    print(f"   Original: {len(X_train)} stars")
    print(f"   Augmented: {len(augmented_data)} stars")
    
    # 8. Visualizations
    print("\n8. Creating visualizations...")
    visualizer = AstrophysicsVisualizer()
    
    # Latent space colored by mass
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(device)
        mu, _ = vae.encode(X_tensor)
        latent = mu.cpu().numpy()
    
    # Mass is first feature (before preprocessing)
    mass_values = stellar_data[:split_idx, 0]
    
    visualizer.plot_latent_space_2d(
        latent,
        labels=mass_values,
        title="Stellar Parameter Latent Space (colored by mass)",
        save_path='results/stellar_latent_2d.png'
    )
    
    # Plot HR diagram evolution (Temperature vs Luminosity)
    sample_evolution = evolutions[0]
    sample_original = preprocessor.inverse_transform(sample_evolution)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot evolution path
    temperatures = sample_original[:, 2]  # Temperature
    luminosities = sample_original[:, 1]  # Luminosity
    
    # HR diagram (inverted temperature axis, log luminosity)
    import matplotlib.pyplot as plt
    ax.plot(temperatures, luminosities, 'o-', linewidth=2, markersize=6, alpha=0.7)
    ax.scatter(temperatures[0], luminosities[0], s=200, c='green', marker='*', 
               edgecolor='black', linewidth=2, label='Initial', zorder=5)
    ax.scatter(temperatures[-1], luminosities[-1], s=200, c='red', marker='X',
               edgecolor='black', linewidth=2, label='Final', zorder=5)
    
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Luminosity (L☉)', fontsize=12)
    ax.set_title('Stellar Evolution Path (Hertzsprung-Russell Diagram)', fontsize=14)
    ax.set_yscale('log')
    ax.invert_xaxis()  # HR diagram convention
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('results/stellar_hr_evolution.png', dpi=300)
    
    # 9. Export
    print("\n9. Exporting results...")
    from visualization.export import AstrophysicsExporter
    
    exporter = AstrophysicsExporter(metadata={
        'domain': 'stellar_evolution',
        'n_constraints': len(physics_domain.constraints),
        'physics_weight': pinn_config.physics_weight
    })
    
    augmented_original = preprocessor.inverse_transform(augmented_data)
    
    exporter.export_to_fits(
        augmented_original,
        'results/stellar_augmented_catalog.fits',
        column_names=physics_domain.feature_names,
        column_units=physics_domain.physical_units,
        header_cards={
            'ORIGIN': 'VAE+PINN Augmentation',
            'NSTARS': len(augmented_original)
        }
    )
    
    print("\n" + "="*60)
    print("Stellar Evolution Example Complete!")
    print("="*60)
    print("\nPhysics-informed augmentation with:")
    print("  - Mass-luminosity relation enforced")
    print("  - Temporal evolution constraints")
    print("  - Physical parameter ranges validated")


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints/stellar', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    main()
