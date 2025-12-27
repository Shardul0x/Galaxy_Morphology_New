# data/augmentation.py
"""
Astrophysics Data Augmentation Pipeline
Generates synthetic data using VAE + PINN
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from models.vae import GenericVAE
from models.pinn import GenericPINN
from config.physics_config import PhysicsDomain

class AstrophysicsAugmentor:
    """
    Data augmentation system for astrophysics research
    
    Combines VAE for feature generation and PINN for temporal evolution
    
    Args:
        vae: Trained VAE model
        pinn: Trained PINN model
        physics_domain: Physics domain configuration
        device: Compute device
    """
    
    def __init__(
        self,
        vae: GenericVAE,
        pinn: GenericPINN,
        physics_domain: PhysicsDomain,
        device: torch.device = torch.device('cpu')
    ):
        self.vae = vae.to(device)
        self.pinn = pinn.to(device)
        self.physics_domain = physics_domain
        self.device = device
        
        # Set models to evaluation mode
        self.vae.eval()
        self.pinn.eval()
    
    def generate_samples(
        self,
        num_samples: int,
        validate_physics: bool = True
    ) -> np.ndarray:
        """
        Generate new synthetic samples
        
        Args:
            num_samples: Number of samples to generate
            validate_physics: Whether to validate against physical constraints
            
        Returns:
            Generated samples [num_samples, feature_dim]
        """
        with torch.no_grad():
            samples = self.vae.sample(num_samples, self.device)
            samples_np = samples.cpu().numpy()
        
        if validate_physics:
            # Filter out physically invalid samples
            valid_samples = []
            for sample in samples_np:
                if self.physics_domain.validate_features(sample):
                    valid_samples.append(sample)
            
            if len(valid_samples) < num_samples:
                print(f"Warning: Only {len(valid_samples)}/{num_samples} samples passed physics validation")
            
            return np.array(valid_samples) if valid_samples else samples_np
        
        return samples_np
    
    def evolve_sample(
        self,
        initial_state: np.ndarray,
        num_steps: int = 50,
        time_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Simulate temporal evolution of a sample
        
        Args:
            initial_state: Initial feature values [feature_dim]
            num_steps: Number of time steps
            time_range: (start_time, end_time)
            
        Returns:
            Evolved states [num_steps, feature_dim]
        """
        self.pinn.eval()
        
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(initial_state).unsqueeze(0).to(self.device)
            
            # Generate time points
            t_values = np.linspace(time_range[0], time_range[1], num_steps)
            
            # Evolve over time
            evolved_states = []
            
            for t_val in t_values:
                t = torch.FloatTensor([[t_val]]).to(self.device)
                evolved = self.pinn(x, t)
                evolved_states.append(evolved.cpu().numpy()[0])
        
        return np.array(evolved_states)
    
    def augment_dataset(
        self,
        original_data: np.ndarray,
        augmentation_factor: int = 2,
        include_evolution: bool = True,
        evolution_steps: int = 10
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Augment an existing dataset
        
        Args:
            original_data: Original dataset [n_samples, feature_dim]
            augmentation_factor: How many times to multiply dataset size
            include_evolution: Whether to generate temporal evolution
            evolution_steps: Number of evolution steps per sample
            
        Returns:
            augmented_data: Augmented dataset
            evolutions: Optional list of temporal evolutions
        """
        n_original = len(original_data)
        n_generate = n_original * (augmentation_factor - 1)
        
        print(f"Augmenting dataset: {n_original} → {n_original + n_generate} samples")
        
        # Generate new samples
        synthetic_samples = self.generate_samples(n_generate)
        
        # Combine with original data
        augmented_data = np.vstack([original_data, synthetic_samples])
        
        # Generate evolution sequences if requested
        evolutions = None
        if include_evolution:
            print(f"Generating temporal evolutions ({evolution_steps} steps per sample)...")
            evolutions = []
            
            for i, sample in enumerate(augmented_data):
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(augmented_data)}")
                
                evolution = self.evolve_sample(sample, num_steps=evolution_steps)
                evolutions.append(evolution)
        
        return augmented_data, evolutions
    
    def interpolate_samples(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Interpolate between two samples in latent space
        
        Args:
            sample1: First sample [feature_dim]
            sample2: Second sample [feature_dim]
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples [num_steps, feature_dim]
        """
        with torch.no_grad():
            x1 = torch.FloatTensor(sample1).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(sample2).unsqueeze(0).to(self.device)
            
            interpolated = self.vae.interpolate(x1, x2, num_steps)
            
        return interpolated.cpu().numpy()
