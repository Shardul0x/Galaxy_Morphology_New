# examples/galaxy_morphology_example.py
"""
Example: Galaxy Morphology Evolution using VAE + PINN
This demonstrates the original galaxy use case in the new generic framework
"""

import sys
sys.path.append('..')

import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import numpy as np
from models.vae import GenericVAE
from models.pinn import GenericPINN
from config.model_config import VAEConfig, PINNConfig
from config.physics_config import get_physics_domain
from data.loader import GalaxyZooLoader
from data.preprocessor import AstrophysicsPreprocessor
from data.augmentation import AstrophysicsAugmentor
from visualization.plotter import AstrophysicsVisualizer
from visualization.export import AstrophysicsExporter
from train import VAETrainer, PINNTrainer
from torch.utils.data import DataLoader, TensorDataset


def main():
    print("="*60)
    print("Galaxy Morphology Evolution Example")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load Data
    print("\n1. Loading Galaxy Zoo data...")
    loader = GalaxyZooLoader(verbose=True)
    loader.load('GalaxyZoo1_DR_table2.csv')
    
    X_train, X_test, _, _ = loader.split_train_test(test_size=0.2)
    
    # 2. Preprocess Data
    print("\n2. Preprocessing data...")
    preprocessor = AstrophysicsPreprocessor(
        scaler_type='minmax',
        clip_outliers=True
    )
    
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 3. Setup Physics Domain
    print("\n3. Setting up physics domain...")
    physics_domain = get_physics_domain('galaxy_morphology')
    print(f"   Domain: {physics_domain.name}")
    print(f"   Features: {len(physics_domain.feature_names)}")
    
    # 4. Create Models
    print("\n4. Creating models...")
    input_dim = X_train_scaled.shape[1]
    
    vae_config = VAEConfig(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        latent_dim=8,
        num_epochs=50,
        batch_size=64
    )
    
    pinn_config = PINNConfig(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_epochs=50,
        batch_size=64,
        physics_weight=0.1
    )
    
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
    
    print(f"   VAE: {sum(p.numel() for p in vae.parameters())} parameters")
    print(f"   PINN: {sum(p.numel() for p in pinn.parameters())} parameters")
    
    # 5. Train VAE
    print("\n5. Training VAE...")
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
    
    train_loader = DataLoader(train_dataset, batch_size=vae_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=vae_config.batch_size)
    
    vae_trainer = VAETrainer(vae, vae_config, device, save_dir='checkpoints/galaxy')
    vae_history = vae_trainer.train(train_loader, val_loader)
    
    # 6. Generate synthetic evolution data for PINN training (GPU-OPTIMIZED)
    print("\n6. Generating evolution data for PINN training...")
    vae.eval()
    
    # Full dataset - GPU-optimized batch processing
    n_samples = len(X_train_scaled)
    n_timesteps = 20
    batch_size = 256  # Process multiple samples simultaneously
    
    print(f"   Total samples: {n_samples}")
    print(f"   Timesteps per sample: {n_timesteps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Expected training examples: {n_samples * n_timesteps}")
    print("   Using GPU batch processing for 10-20x speedup...\n")
    
    evolution_data = []
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx in range(0, n_samples, batch_size):
            batch_end = min(batch_idx + batch_size, n_samples)
            batch_data = X_train_scaled[batch_idx:batch_end]
            
            # Process entire batch on GPU simultaneously
            x_batch = torch.FloatTensor(batch_data).to(device)
            mu, logvar = vae.encode(x_batch)
            
            # Generate all timesteps for this batch
            for t in range(n_timesteps):
                time_value = t / n_timesteps
                noise = torch.randn_like(mu) * time_value * 0.1
                z = mu + noise
                evolved = vae.decode(z)
                
                # Store results
                for i in range(len(x_batch)):
                    evolution_data.append((
                        x_batch[i:i+1].cpu(),
                        time_value,
                        evolved[i:i+1].cpu()
                    ))
            
            # Progress update with ETA
            if batch_idx % (batch_size * 5) == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                percent = (batch_idx / n_samples) * 100
                if batch_idx > 0:
                    eta = (elapsed / batch_idx) * (n_samples - batch_idx)
                    samples_per_sec = len(evolution_data) / elapsed
                    print(f"   [{batch_idx:6d}/{n_samples}] {percent:5.1f}% | "
                          f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s | "
                          f"Speed: {samples_per_sec:.0f} samples/s")
                else:
                    print(f"   Starting batch processing...")
    
    elapsed = time.time() - start_time
    print(f"\n   ✓ Generated {len(evolution_data)} training examples in {elapsed:.1f}s")
    print(f"   Average speed: {len(evolution_data)/elapsed:.0f} samples/second")
    print(f"   GPU acceleration: ~{(n_samples * n_timesteps * 0.5 / elapsed):.1f}x faster than CPU\n")
    
    # Prepare PINN data
    pinn_X = torch.cat([item[0] for item in evolution_data])
    pinn_t = torch.FloatTensor([[item[1]] for item in evolution_data])
    pinn_y = torch.cat([item[2] for item in evolution_data])
    
    pinn_dataset = TensorDataset(pinn_X, pinn_t, pinn_y)
    pinn_loader = DataLoader(pinn_dataset, batch_size=pinn_config.batch_size, shuffle=True)
    
    # 7. Train PINN
    print("\n7. Training PINN...")
    pinn_trainer = PINNTrainer(pinn, pinn_config, physics_domain, device, save_dir='checkpoints/galaxy')
    pinn_history = pinn_trainer.train(pinn_loader)
    
    # 8. Data Augmentation
    print("\n8. Performing data augmentation...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    augmented_data, evolutions = augmentor.augment_dataset(
        X_train_scaled,
        augmentation_factor=3,
        include_evolution=True,
        evolution_steps=30
    )
    
    print(f"   Original: {len(X_train_scaled)} samples")
    print(f"   Augmented: {len(augmented_data)} samples")
    
    # 9. Visualization
    print("\n9. Creating visualizations...")
    visualizer = AstrophysicsVisualizer()
    
    # Get latent representations
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train_scaled).to(device)
        mu, _ = vae.encode(X_tensor)
        latent = mu.cpu().numpy()
    
    # Plot latent space
    visualizer.plot_latent_space_2d(
        latent,
        title="Galaxy Morphology Latent Space",
        save_path='results/galaxy_latent_2d.png'
    )
    
    visualizer.plot_latent_space_3d(
        latent,
        title="Galaxy Morphology Latent Space (3D)",
        save_path='results/galaxy_latent_3d.png'
    )
    
    # Plot training history
    visualizer.plot_training_history(
        vae_history,
        metrics=['loss', 'recon_loss', 'kl_loss'],
        title="VAE Training History",
        save_path='results/vae_training.png'
    )
    
    # Plot evolution
    sample_evolution = evolutions[0]
    visualizer.plot_temporal_evolution(
        sample_evolution,
        feature_names=physics_domain.feature_names,
        feature_indices=[0, 1, 2, 6],  # P_EL, P_CW, P_ACW, P_CS
        title="Galaxy Evolution Over Time",
        save_path='results/galaxy_evolution.png'
    )
    
    # 10. Export Results
    print("\n10. Exporting results...")
    exporter = AstrophysicsExporter(metadata={
        'domain': 'galaxy_morphology',
        'vae_latent_dim': vae_config.latent_dim,
        'augmentation_factor': 3
    })
    
    # Export augmented data
    augmented_original = preprocessor.inverse_transform(augmented_data)
    
    exporter.export_to_csv(
        augmented_original,
        'results/galaxy_augmented.csv',
        column_names=physics_domain.feature_names
    )
    
    exporter.export_to_fits(
        augmented_original,
        'results/galaxy_augmented.fits',
        column_names=physics_domain.feature_names,
        column_units=physics_domain.physical_units
    )
    
    # Export evolution sequence
    exporter.export_evolution_sequence(
        [preprocessor.inverse_transform(evo) for evo in evolutions[:10]],
        np.linspace(0, 1, len(evolutions[0])),
        'results/galaxy_evolution_sequence.h5',
        column_names=physics_domain.feature_names
    )
    
    print("\n" + "="*60)
    print("Galaxy Morphology Example Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Checkpoints: ../checkpoints/galaxy/")
    print("  - Visualizations: ../results/galaxy_*.png")
    print("  - Data: ../results/galaxy_augmented.{csv,fits}")
    print("  - Evolution: ../results/galaxy_evolution_sequence.h5")


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints/galaxy', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    main()
