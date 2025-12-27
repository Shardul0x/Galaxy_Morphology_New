# galaxy_example_fixed.py
# Fixed Galaxy Morphology Example with Proper PINN Training

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    print("Galaxy Morphology Evolution Example (FIXED)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load Data
    print("\n1. Loading Galaxy Zoo data...")
    loader = GalaxyZooLoader(verbose=True)
    loader.load('GalaxyZoo1_DR_table2.csv')
    
    X_train, X_test, _, _ = loader.split_train_test(test_size=0.2)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 2. Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = AstrophysicsPreprocessor(
        scaler_type='minmax',
        clip_outliers=True
    )
    
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 3. Setup Physics
    print("\n3. Setting up physics domain...")
    physics_domain = get_physics_domain('galaxy_morphology')
    
    # 4. Create Models
    print("\n4. Creating models...")
    input_dim = X_train_scaled.shape[1]
    
    vae_config = VAEConfig(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        latent_dim=8,
        num_epochs=30,
        batch_size=128
    )
    
    pinn_config = PINNConfig(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_epochs=30,
        batch_size=128,
        physics_weight=0.01
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
    
    # 5. Train VAE
    print("\n5. Training VAE...")
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled)),
        batch_size=vae_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled)),
        batch_size=vae_config.batch_size,
        num_workers=0
    )
    
    vae_trainer = VAETrainer(vae, vae_config, device, save_dir='checkpoints/galaxy_fixed')
    vae_history = vae_trainer.train(train_loader, val_loader)
    
    # 6. Generate PROPER evolution data
    print("\n6. Generating evolution data for PINN (FIXED VERSION)...")
    vae.eval()
    
    n_samples = min(5000, len(X_train_scaled))
    n_timesteps = 10
    batch_size = 256
    
    print(f"   Samples: {n_samples}, Timesteps: {n_timesteps}, Batch: {batch_size}")
    
    pinn_data_X = []
    pinn_data_t = []
    pinn_data_y = []
    
    with torch.no_grad():
        for batch_idx in range(0, n_samples, batch_size):
            batch_end = min(batch_idx + batch_size, n_samples)
            x_batch = torch.FloatTensor(X_train_scaled[batch_idx:batch_end]).to(device)
            
            mu, logvar = vae.encode(x_batch)
            
            for t_idx in range(n_timesteps):
                time_value = t_idx / (n_timesteps - 1)
                
                # Add evolution noise
                noise_scale = 0.2 * time_value
                z = mu + torch.randn_like(mu) * noise_scale
                evolved = vae.decode(z)
                
                # Add observation noise
                evolved = evolved + torch.randn_like(evolved) * 0.01
                
                pinn_data_X.append(x_batch.cpu())
                pinn_data_t.append(torch.full((len(x_batch), 1), time_value))
                pinn_data_y.append(evolved.cpu())
            
            if batch_idx % 1000 == 0:
                print(f"   Progress: {batch_idx}/{n_samples}")
    
    pinn_X = torch.cat(pinn_data_X)
    pinn_t = torch.cat(pinn_data_t)
    pinn_y = torch.cat(pinn_data_y)
    
    print(f"   OK Generated {len(pinn_X)} training examples")
    print(f"   Data range check:")
    print(f"     X: [{pinn_X.min():.4f}, {pinn_X.max():.4f}]")
    print(f"     y: [{pinn_y.min():.4f}, {pinn_y.max():.4f}]")
    
    # 7. Train PINN with PROPER batch size
    print("\n7. Training PINN (FIXED VERSION)...")
    pinn_loader = DataLoader(
        TensorDataset(pinn_X, pinn_t, pinn_y),
        batch_size=pinn_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Batches per epoch: {len(pinn_loader)}")
    print(f"   This should be ~400, not 166,000!")
    
    pinn_trainer = PINNTrainer(pinn, pinn_config, physics_domain, device, save_dir='checkpoints/galaxy_fixed')
    pinn_history = pinn_trainer.train(pinn_loader)
    
    # 8. Augment
    print("\n8. Augmenting dataset...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    augmented_data, evolutions = augmentor.augment_dataset(
        X_train_scaled[:1000],
        augmentation_factor=3,
        include_evolution=True,
        evolution_steps=20
    )
    
    print(f"   Generated {len(augmented_data)} samples")
    
    # 9. Visualize
    print("\n9. Creating visualizations...")
    visualizer = AstrophysicsVisualizer()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train_scaled[:5000]).to(device)
        mu, _ = vae.encode(X_tensor)
        latent = mu.cpu().numpy()
    
    visualizer.plot_latent_space_2d(
        latent,
        title="Galaxy Latent Space (Fixed)",
        save_path='results/galaxy_fixed_latent.png'
    )
    
    # 10. Export
    print("\n10. Exporting results...")
    exporter = AstrophysicsExporter()
    augmented_original = preprocessor.inverse_transform(augmented_data)
    
    exporter.export_to_csv(
        augmented_original,
        'results/galaxy_fixed_augmented.csv',
        column_names=physics_domain.feature_names
    )
    
    print("\n" + "="*60)
    print("OK COMPLETE (Fixed Version)!")
    print("="*60)


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints/galaxy_fixed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()
