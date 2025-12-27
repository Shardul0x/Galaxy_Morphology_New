# run_pipeline.py - FIXED VERSION
"""Simplified pipeline runner with correct path handling"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, TensorDataset

from models.vae import GenericVAE
from models.pinn import GenericPINN
from data.loader import GalaxyZooLoader
from data.preprocessor import AstrophysicsPreprocessor
from data.augmentation import AstrophysicsAugmentor
from config.physics_config import get_physics_domain
from train import VAETrainer, PINNTrainer
from config.model_config import VAEConfig, PINNConfig


def run_augmentation_pipeline(
    data_path: str,
    domain: str = "galaxy_morphology",
    augmentation_factor: int = 3,
    output_dir: str = "results",
    epochs: int = 20
):
    """Run complete augmentation pipeline"""
    
    print("="*70)
    print(f"ASTROPHYSICS DATA AUGMENTATION PIPELINE")
    print(f"Domain: {domain}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load config
    with open('config/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    domain_config = config['domains'][domain]
    defaults = config['defaults']
    
    # Create directories - FIX PATH SEPARATOR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_path / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput: {output_path}")
    print(f"Checkpoints: {checkpoint_path}")
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    loader = GalaxyZooLoader(verbose=False)
    loader.load(data_path)
    X_train, X_test, _, _ = loader.split_train_test(test_size=0.2)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # 2. Preprocess
    print("\n[2/6] Preprocessing...")
    preprocessor = AstrophysicsPreprocessor(scaler_type=defaults['normalization'])
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 3. Setup Models
    print("\n[3/6] Building models...")
    input_dim = X_train_scaled.shape[1]
    
    vae = GenericVAE(
        input_dim=input_dim,
        hidden_dims=domain_config['vae_dims'],
        latent_dim=domain_config['latent_dim']
    ).to(device)
    
    physics_domain = get_physics_domain(domain)
    pinn = GenericPINN(
        input_dim=input_dim,
        hidden_dims=domain_config['pinn_dims'],
        physics_domain=physics_domain
    ).to(device)
    
    print(f"  VAE: {sum(p.numel() for p in vae.parameters()):,} params")
    print(f"  PINN: {sum(p.numel() for p in pinn.parameters()):,} params")
    
    # 4. Train VAE
    print(f"\n[4/6] Training VAE ({epochs} epochs)...")
    vae_config = VAEConfig(
        input_dim=input_dim,
        hidden_dims=domain_config['vae_dims'],
        latent_dim=domain_config['latent_dim'],
        num_epochs=epochs,
        batch_size=defaults['batch_size']
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled)),
        batch_size=defaults['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled)),
        batch_size=defaults['batch_size'],
        num_workers=0
    )
    
    vae_trainer = VAETrainer(vae, vae_config, device, save_dir=str(checkpoint_path))
    vae_trainer.train(train_loader, val_loader)
    
    # 5. Generate PINN data & Train
    print(f"\n[5/6] Training PINN ({epochs} epochs)...")
    vae.eval()
    
    # Quick evolution data generation
    n_samples = min(2000, len(X_train_scaled))
    pinn_X, pinn_t, pinn_y = [], [], []
    
    print(f"  Generating evolution data for {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(0, n_samples, 256):
            batch_end = min(i + 256, n_samples)
            batch = torch.FloatTensor(X_train_scaled[i:batch_end]).to(device)
            mu, _ = vae.encode(batch)
            
            for t in range(10):
                time_val = t / 9
                z = mu + torch.randn_like(mu) * 0.2 * time_val
                evolved = vae.decode(z) + torch.randn_like(batch) * 0.01
                
                pinn_X.append(batch.cpu())
                pinn_t.append(torch.full((len(batch), 1), time_val))
                pinn_y.append(evolved.cpu())
    
    pinn_loader = DataLoader(
        TensorDataset(torch.cat(pinn_X), torch.cat(pinn_t), torch.cat(pinn_y)),
        batch_size=defaults['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    pinn_config = PINNConfig(
        input_dim=input_dim,
        hidden_dims=domain_config['pinn_dims'],
        num_epochs=epochs,
        batch_size=defaults['batch_size'],
        physics_weight=defaults['physics_weight']
    )
    
    pinn_trainer = PINNTrainer(pinn, pinn_config, physics_domain, device, save_dir=str(checkpoint_path))
    pinn_trainer.train(pinn_loader)
    
    # 6. Augment & Export
    print(f"\n[6/6] Generating {augmentation_factor}x augmented data...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    augmented, _ = augmentor.augment_dataset(
        X_train_scaled[:1000],
        augmentation_factor=augmentation_factor,
        include_evolution=False
    )
    
    augmented_original = preprocessor.inverse_transform(augmented)
    
    # Export - FIX PATH
    output_csv = output_path / 'augmented_data.csv'
    df = pd.DataFrame(augmented_original, columns=domain_config['features'])
    df.to_csv(str(output_csv), index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated: {len(augmented):,} samples")
    print(f"Saved to: {output_csv}")
    print(f"Models: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    return {
        'output_csv': str(output_csv),
        'samples': len(augmented),
        'checkpoint_dir': str(checkpoint_path)
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <data.csv> [domain] [augment_factor]")
        print("Example: python run_pipeline.py GalaxyZoo1_DR_table2.csv galaxy_morphology 3")
        sys.exit(1)
    
    data_path = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "galaxy_morphology"
    aug_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    run_augmentation_pipeline(data_path, domain, aug_factor)
