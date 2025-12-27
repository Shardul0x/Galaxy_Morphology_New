# models/pinn.py
"""
Generic Physics-Informed Neural Network for Temporal Evolution
Learns time-dependent dynamics constrained by physical laws
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable
from config.physics_config import PhysicsDomain

class GenericPINN(nn.Module):
    """
    Physics-Informed Neural Network for astrophysics simulations
    
    Predicts temporal evolution of features while respecting physical constraints
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        physics_domain: Optional PhysicsDomain for constraint enforcement
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64, 32],
        activation: str = 'tanh',
        physics_domain: Optional[PhysicsDomain] = None
    ):
        super(GenericPINN, self).__init__()
        
        self.input_dim = input_dim
        self.physics_domain = physics_domain
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build network: input = [features, time] -> output = features
        layers = []
        prev_dim = input_dim + 1  # +1 for time dimension
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation.lower()]
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict evolved features at time t
        
        Args:
            x: Input features [batch_size, input_dim]
            t: Time values [batch_size, 1] or [batch_size]
            
        Returns:
            Evolved features [batch_size, input_dim]
        """
        # Ensure time is 2D
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate features and time
        x_t = torch.cat([x, t], dim=1)
        
        # Predict evolved state
        evolved = self.network(x_t)
        
        return evolved
    
    def compute_time_derivative(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute time derivative dx/dt using automatic differentiation
        
        Args:
            x: Input features [batch_size, input_dim]
            t: Time values [batch_size, 1], requires_grad=True
            
        Returns:
            Time derivative dx/dt [batch_size, input_dim]
        """
        # Enable gradient computation
        t.requires_grad_(True)
        
        # Forward pass
        evolved = self.forward(x, t)
        
        # Compute gradients
        gradients = []
        for i in range(evolved.size(1)):
            grad = torch.autograd.grad(
                evolved[:, i].sum(),
                t,
                create_graph=True,
                retain_graph=True
            )[0]
            gradients.append(grad)
        
        # Stack gradients
        dx_dt = torch.cat(gradients, dim=1)
        
        return dx_dt


def pinn_physics_loss(
    model: GenericPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    physics_domain: PhysicsDomain
) -> torch.Tensor:
    """
    Compute physics-based loss from domain constraints
    
    Args:
        model: PINN model
        x: Input features
        t: Time values
        physics_domain: Physics domain with constraints
        
    Returns:
        Total physics loss
    """
    if not physics_domain.constraints:
        return torch.tensor(0.0, device=x.device)
    
    # Get model predictions
    evolved = model(x, t)
    
    # Compute time derivative
    dx_dt = model.compute_time_derivative(x, t)
    
    # Accumulate constraint losses
    total_physics_loss = torch.tensor(0.0, device=x.device)
    
    for constraint in physics_domain.constraints:
        constraint_loss = constraint.equation(evolved, dx_dt)
        total_physics_loss += constraint.weight * constraint_loss
    
    return total_physics_loss


def pinn_total_loss(
    model: GenericPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    x_target: torch.Tensor,
    physics_domain: Optional[PhysicsDomain] = None,
    physics_weight: float = 0.1
) -> tuple:
    """
    Compute total PINN loss = Data Loss + Physics Loss
    
    Args:
        model: PINN model
        x: Input features
        t: Time values
        x_target: Target features at time t
        physics_domain: Optional physics domain for constraints
        physics_weight: Weight for physics loss
        
    Returns:
        total_loss, data_loss, physics_loss
    """
    # Data fitting loss (MSE)
    evolved = model(x, t)
    data_loss = nn.functional.mse_loss(evolved, x_target)
    
    # Physics constraint loss
    if physics_domain is not None and physics_domain.constraints:
        physics_loss = pinn_physics_loss(model, x, t, physics_domain)
    else:
        physics_loss = torch.tensor(0.0, device=x.device)
    
    # Total loss
    total_loss = data_loss + physics_weight * physics_loss
    
    return total_loss, data_loss, physics_loss
