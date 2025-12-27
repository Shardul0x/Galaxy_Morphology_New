# debug_pinn.py
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

print(f"\nTest data shapes:")
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

print(f"\nModel created: {sum(p.numel() for p in pinn.parameters())} parameters")

# Test forward pass
print("\nTesting forward pass...")
y_pred = pinn(X, t)
print(f"  OK Forward pass: {y_pred.shape}")
print(f"  Output range: [{y_pred.min().item():.4f}, {y_pred.max().item():.4f}]")

# Test loss computation
print("\nTesting loss computation...")

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
print("\nTesting backward pass...")
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
    print("\nX PROBLEM: Gradients are vanishing!")
    print("   Possible causes:")
    print("   1. Learning rate too small")
    print("   2. Wrong loss computation")
    print("   3. Data normalization issue")
else:
    print("\nOK Gradients look okay")

# Test training step
optimizer.step()
print("\nOK Training step successful")

print("\n" + "="*60)
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
