# models/vae.py
"""
Generic Variational Autoencoder for Astrophysics Data
Flexible architecture that adapts to different feature dimensions
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class GenericVAE(nn.Module):
    """
    Flexible VAE architecture for astrophysics data augmentation
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions for encoder
        latent_dim: Dimension of latent space
        activation: Activation function ('relu', 'leakyrelu', 'tanh', 'elu')
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16],
        latent_dim: int = 8,
        activation: str = 'leakyrelu',
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super(GenericVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            encoder_layers.append(self.activation)
            
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1]
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            
            if use_batch_norm and i < len(decoder_dims) - 2:
                decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
            decoder_layers.append(self.activation)
            
            if dropout_rate > 0 and i < len(decoder_dims) - 2:
                decoder_layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output normalized to [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        
        return activations[activation.lower()]
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to feature space
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed features
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            reconstruction: Reconstructed features
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new samples by sampling from latent space
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two samples in latent space
        
        Args:
            x1: First sample [1, input_dim]
            x2: Second sample [1, input_dim]
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples [num_steps, input_dim]
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps).unsqueeze(1).to(x1.device)
        z_interp = mu1 + alphas * (mu2 - mu1)
        
        interpolated = self.decode(z_interp)
        return interpolated


def vae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        reconstruction: Reconstructed features
        original: Original features
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (β-VAE)
        
    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
    
    # KL divergence loss
    # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / original.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
