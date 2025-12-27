# test_installation.py
"""Quick test to verify installation"""

import torch
import numpy as np
from config.physics_config import AVAILABLE_DOMAINS, get_physics_domain
from models.vae import GenericVAE
from models.pinn import GenericPINN

print("="*60)
print("Testing Astrophysics Augmentation Pipeline")
print("="*60)

# Test 1: Check PyTorch
print("\n✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ CUDA device:", torch.cuda.get_device_name(0))

# Test 2: Check available domains
print("\n✓ Available physics domains:")
for domain in AVAILABLE_DOMAINS.keys():
    print(f"  - {domain}")

# Test 3: Load a physics domain
print("\n✓ Testing Galaxy Morphology domain:")
domain = get_physics_domain('galaxy_morphology')
print(f"  Features: {len(domain.feature_names)}")
print(f"  Constraints: {len(domain.constraints)}")

# Test 4: Create models
print("\n✓ Creating models:")
vae = GenericVAE(input_dim=9, hidden_dims=[64, 32], latent_dim=8)
pinn = GenericPINN(input_dim=9, hidden_dims=[64, 32])
print(f"  VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
print(f"  PINN parameters: {sum(p.numel() for p in pinn.parameters()):,}")

print("\n" + "="*60)
print("✓ All tests passed! Ready to use.")
print("="*60)
