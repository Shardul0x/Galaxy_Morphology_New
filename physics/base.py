# physics/base.py
"""
Base classes for physics constraints and equations
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Any
from abc import ABC, abstractmethod


class PhysicsEquation(ABC):
    """
    Abstract base class for physics equations
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the physics equation
        
        Args:
            features: Input features [batch_size, n_features]
            **kwargs: Additional parameters
            
        Returns:
            Computed value or residual
        """
        pass
    
    def loss(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss based on physics equation
        
        Args:
            features: Input features
            **kwargs: Additional parameters
            
        Returns:
            Physics loss (scalar)
        """
        residual = self.compute(features, **kwargs)
        return torch.mean(residual ** 2)


class ConservationLaw(PhysicsEquation):
    """
    Base class for conservation laws (energy, momentum, etc.)
    """
    
    def __init__(self, name: str, conserved_quantity_fn: Callable, description: str = ""):
        super().__init__(name, description)
        self.conserved_quantity_fn = conserved_quantity_fn
    
    def compute(self, features: torch.Tensor, time_derivative: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check if quantity is conserved (derivative = 0)
        
        Args:
            features: Input features
            time_derivative: Optional time derivatives
            
        Returns:
            Deviation from conservation
        """
        if time_derivative is not None:
            # Compute rate of change of conserved quantity
            conserved = self.conserved_quantity_fn(features)
            d_conserved = self.conserved_quantity_fn(features + time_derivative * 0.01) - conserved
            return d_conserved
        else:
            # Check variance across batch (should be constant)
            conserved = self.conserved_quantity_fn(features)
            return conserved - conserved.mean()


class PowerLaw(PhysicsEquation):
    """
    Generic power law relation: Y = k * X^alpha
    """
    
    def __init__(
        self,
        name: str,
        x_index: int,
        y_index: int,
        alpha: float,
        k: float = 1.0,
        description: str = ""
    ):
        super().__init__(name, description)
        self.x_index = x_index
        self.y_index = y_index
        self.alpha = alpha
        self.k = k
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute residual: Y - k * X^alpha
        
        Args:
            features: Input features
            
        Returns:
            Residual
        """
        x = features[:, self.x_index]
        y = features[:, self.y_index]
        
        expected_y = self.k * (x ** self.alpha)
        residual = y - expected_y
        
        return residual


class LinearRelation(PhysicsEquation):
    """
    Generic linear relation: Y = a * X + b
    """
    
    def __init__(
        self,
        name: str,
        x_index: int,
        y_index: int,
        slope: float,
        intercept: float = 0.0,
        description: str = ""
    ):
        super().__init__(name, description)
        self.x_index = x_index
        self.y_index = y_index
        self.slope = slope
        self.intercept = intercept
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute residual: Y - (a * X + b)
        
        Args:
            features: Input features
            
        Returns:
            Residual
        """
        x = features[:, self.x_index]
        y = features[:, self.y_index]
        
        expected_y = self.slope * x + self.intercept
        residual = y - expected_y
        
        return residual


class RangeConstraint(PhysicsEquation):
    """
    Enforce that features stay within physical ranges
    """
    
    def __init__(
        self,
        name: str,
        feature_index: int,
        min_value: float,
        max_value: float,
        description: str = ""
    ):
        super().__init__(name, description)
        self.feature_index = feature_index
        self.min_value = min_value
        self.max_value = max_value
    
    def compute(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute violation of range constraints
        
        Args:
            features: Input features
            
        Returns:
            Penalty for being outside range
        """
        x = features[:, self.feature_index]
        
        # Penalty for being below minimum
        below_penalty = torch.relu(self.min_value - x)
        
        # Penalty for being above maximum
        above_penalty = torch.relu(x - self.max_value)
        
        return below_penalty + above_penalty
