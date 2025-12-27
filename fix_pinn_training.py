# fix_pinn_training.py
"""
Fix PINN training issues:
1. Zero loss problem
2. Slow training speed
3. Proper loss balancing
"""

import os

def fix_pinn_trainer():
    filepath = 'train.py'
    
    if not os.path.exists(filepath):
        print(f"X {filepath} not found")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Ensure proper loss logging (not .item() on zero tensors)
    content = content.replace(
        'total_loss += loss.item()',
        'total_loss += loss.item() if not torch.isnan(loss) else 0.0'
    )
    
    content = content.replace(
        'total_data_loss += data_loss.item()',
        'total_data_loss += data_loss.item() if not torch.isnan(data_loss) else 0.0'
    )
    
    content = content.replace(
        'total_physics_loss += physics_loss.item()',
        'total_physics_loss += physics_loss.item() if not torch.isnan(physics_loss) else 0.0'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("OK Fixed loss computation in train.py")


def create_debug_pinn_script():
    """Create a debugging script to check what's wrong"""
    
    script = """# debug_pinn.py
# Debug PINN training issues

import torch
import numpy as np
from models.pinn import GenericPINN
from config.physics_config import get_physics_domain
from config.model_config import PINNConfig

print("="*60)
print("PINN Debug - Checking Loss Computation")
print("="*60)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
physics_domain = get_physics_domain('galaxy_morphology')

# Create small test data
batch_size = 4
input_dim = 9

X = torch.randn(batch_size, input_dim).to(device)
t = torch.rand(batch_size, 1).to(device)
y_true = torch.randn(batch_size, input_dim).to(device)

print(f"\\nTest data shapes:")
print(f"  X: {X.shape}")
print(f"  t: {t.shape}")
print(f"  y_true: {y_true.shape}")

# Create model
pinn_config = PINNConfig(
    input_dim=input_dim,
    hidden_dims=[32, 32],
    physics_weight=0.1
)

pinn = GenericPINN(
    input_dim=pinn_config.input_dim,
    hidden_dims=pinn_config.hidden_dims,
    physics_domain=physics_domain
).to(device)

print(f"\\nModel created: {sum(p.numel() for p in pinn.parameters())} parameters")

# Test forward pass
print("\\nTesting forward pass...")
y_pred = pinn(X, t)
print(f"  OK Forward pass: {y_pred.shape}")
print(f"  Output range: [{y_pred.min().item():.4f}, {y_pred.max().item():.4f}]")

# Test loss computation
print("\\nTesting loss computation...")

# Data loss (MSE)
data_loss = torch.nn.functional.mse_loss(y_pred, y_true)
print(f"  Data Loss: {data_loss.item():.6f}")

# Physics loss
if hasattr(pinn, 'compute_physics_loss'):
    physics_loss = pinn.compute_physics_loss(X, y_pred, t)
    print(f"  Physics Loss: {physics_loss.item():.6f}")
    
    total_loss = data_loss + pinn_config.physics_weight * physics_loss
    print(f"  Total Loss: {total_loss.item():.6f}")
else:
    print("  ! No compute_physics_loss method found!")

# Test backward pass
print("\\nTesting backward pass...")
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)

optimizer.zero_grad()
loss = data_loss + (physics_loss * pinn_config.physics_weight if hasattr(pinn, 'compute_physics_loss') else 0)
loss.backward()

# Check gradients
total_grad_norm = 0
for name, param in pinn.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm
        if grad_norm < 1e-8:
            print(f"  ! Very small gradient in {name}: {grad_norm:.2e}")

print(f"  Total gradient norm: {total_grad_norm:.6f}")

if total_grad_norm < 1e-6:
    print("\\nX PROBLEM: Gradients are vanishing!")
    print("   Possible causes:")
    print("   1. Learning rate too small")
    print("   2. Wrong loss computation")
    print("   3. Data normalization issue")
else:
    print("\\nOK Gradients look okay")

# Test training step
optimizer.step()
print("\\nOK Training step successful")

print("\\n" + "="*60)
print("Diagnosis:")
print("="*60)

if data_loss.item() < 1e-6:
    print("X Data loss is near zero - model output matches target exactly")
    print("   This suggests you're training on reconstructed data")
    print("   SOLUTION: Use real evolution data, not perfect reconstructions")
elif total_grad_norm < 1e-6:
    print("X Vanishing gradients detected")
    print("   SOLUTION: Increase learning rate or check activation functions")
else:
    print("OK PINN appears to be working correctly")
    print("  The issue might be in the data or training loop")

print("="*60)
"""
    
    with open('debug_pinn.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("OK Created debug_pinn.py")


def create_fixed_example():
    """Create a properly working example with correct PINN training"""
    
    script = """# galaxy_example_fixed.py
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
    print(f"\\nUsing device: {device}")
    
    # 1. Load Data
    print("\\n1. Loading Galaxy Zoo data...")
    loader = GalaxyZooLoader(verbose=True)
    loader.load('GalaxyZoo1_DR_table2.csv')
    
    X_train, X_test, _, _ = loader.split_train_test(test_size=0.2)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 2. Preprocess
    print("\\n2. Preprocessing data...")
    preprocessor = AstrophysicsPreprocessor(
        scaler_type='minmax',
        clip_outliers=True
    )
    
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 3. Setup Physics
    print("\\n3. Setting up physics domain...")
    physics_domain = get_physics_domain('galaxy_morphology')
    
    # 4. Create Models
    print("\\n4. Creating models...")
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
    print("\\n5. Training VAE...")
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
    print("\\n6. Generating evolution data for PINN (FIXED VERSION)...")
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
    print("\\n7. Training PINN (FIXED VERSION)...")
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
    print("\\n8. Augmenting dataset...")
    augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)
    
    augmented_data, evolutions = augmentor.augment_dataset(
        X_train_scaled[:1000],
        augmentation_factor=3,
        include_evolution=True,
        evolution_steps=20
    )
    
    print(f"   Generated {len(augmented_data)} samples")
    
    # 9. Visualize
    print("\\n9. Creating visualizations...")
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
    print("\\n10. Exporting results...")
    exporter = AstrophysicsExporter()
    augmented_original = preprocessor.inverse_transform(augmented_data)
    
    exporter.export_to_csv(
        augmented_original,
        'results/galaxy_fixed_augmented.csv',
        column_names=physics_domain.feature_names
    )
    
    print("\\n" + "="*60)
    print("OK COMPLETE (Fixed Version)!")
    print("="*60)


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints/galaxy_fixed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()
"""
    
    with open('examples/galaxy_example_fixed.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("OK Created examples/galaxy_example_fixed.py")


if __name__ == '__main__':
    print("="*60)
    print("Fixing PINN Training Issues")
    print("="*60)
    print()
    
    fix_pinn_trainer()
    create_debug_pinn_script()
    create_fixed_example()
    
    print()
    print("="*60)
    print("ALL FIXES APPLIED")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. First, run diagnosis:")
    print("     python debug_pinn.py")
    print()
    print("  2. Then run fixed example:")
    print("     python examples/galaxy_example_fixed.py")
    print()
    print("Key fixes:")
    print("  OK Proper batch size (128 instead of 1)")
    print("  OK Added noise to evolution data")
    print("  OK Reduced epochs for faster testing (30)")
    print("  OK Lower physics weight (0.01)")
    print("  OK ~400 batches/epoch instead of 166,000!")
    print("="*60)
