# train.py
"""
Training Pipeline for VAE and PINN Models
Supports logging, checkpointing, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import json
from datetime import datetime

from models.vae import GenericVAE
from models.pinn import GenericPINN, pinn_total_loss
from models.losses import (
    vae_loss_standard,
    vae_loss_weighted_features,
    get_beta_schedule
)
from config.model_config import VAEConfig, PINNConfig
from config.physics_config import PhysicsDomain


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class VAETrainer:
    """Trainer for VAE model"""
    
    def __init__(
        self,
        model: GenericVAE,
        config: VAEConfig,
        device: torch.device,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        
        # Setup scheduler
        if config.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        elif config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        # Get beta value for this epoch
        if self.config.beta_schedule:
            beta = get_beta_schedule(
                epoch, self.config.num_epochs,
                self.config.beta_schedule,
                self.config.beta
            )
        else:
            beta = self.config.beta
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch in pbar:
            if isinstance(batch, list):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Forward pass
            reconstruction, mu, logvar = self.model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss_standard(
                reconstruction, x, mu, logvar, beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{beta:.3f}'
            })
        
        # Average losses
        num_batches = len(train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'kl_loss': epoch_kl_loss / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        # Get beta value
        if self.config.beta_schedule:
            beta = get_beta_schedule(
                epoch, self.config.num_epochs,
                self.config.beta_schedule,
                self.config.beta
            )
        else:
            beta = self.config.beta
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                reconstruction, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = vae_loss_standard(
                    reconstruction, x, mu, logvar, beta
                )
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
        
        num_batches = len(val_loader)
        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'kl_loss': epoch_kl_loss / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Complete training loop"""
        print(f"\n{'='*60}")
        print(f"Training VAE Model")
        print(f"{'='*60}")
        print(f"Input dim: {self.config.input_dim}")
        print(f"Latent dim: {self.config.latent_dim}")
        print(f"Hidden dims: {self.config.hidden_dims}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Beta: {self.config.beta}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = self.validate_epoch(val_loader, epoch)
            else:
                val_metrics = train_metrics
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f})")
            if val_loader:
                print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                      f"(Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f})")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_vae.pth', epoch, val_metrics)
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model and history
        self.save_checkpoint('final_vae.pth', epoch, val_metrics)
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history"""
        filepath = os.path.join(self.save_dir, 'vae_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class PINNTrainer:
    """Trainer for PINN model"""
    
    def __init__(
        self,
        model: GenericPINN,
        config: PINNConfig,
        physics_domain: Optional[PhysicsDomain],
        device: torch.device,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.config = config
        self.physics_domain = physics_domain
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        
        # Setup scheduler
        if config.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        elif config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch in pbar:
            x, t, x_target = batch
            x = x.to(self.device)
            t = t.to(self.device)
            x_target = x_target.to(self.device)
            
            # Compute loss
            loss, data_loss, physics_loss = pinn_total_loss(
                self.model, x, t, x_target,
                self.physics_domain,
                self.config.physics_weight
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'data': f'{data_loss.item():.4f}',
                'physics': f'{physics_loss.item():.4f}'
            })
        
        num_batches = len(train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'data_loss': epoch_data_loss / num_batches,
            'physics_loss': epoch_physics_loss / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x, t, x_target = batch
                x = x.to(self.device)
                t = t.to(self.device)
                x_target = x_target.to(self.device)
                
                loss, data_loss, physics_loss = pinn_total_loss(
                    self.model, x, t, x_target,
                    self.physics_domain,
                    self.config.physics_weight
                )
                
                epoch_loss += loss.item()
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()
        
        num_batches = len(val_loader)
        return {
            'loss': epoch_loss / num_batches,
            'data_loss': epoch_data_loss / num_batches,
            'physics_loss': epoch_physics_loss / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Complete training loop"""
        print(f"\n{'='*60}")
        print(f"Training PINN Model")
        print(f"{'='*60}")
        print(f"Input dim: {self.config.input_dim}")
        print(f"Hidden dims: {self.config.hidden_dims}")
        print(f"Physics domain: {self.physics_domain.name if self.physics_domain else 'None'}")
        print(f"Physics weight: {self.config.physics_weight}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = train_metrics
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_data_loss'].append(train_metrics['data_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_data_loss'].append(val_metrics['data_loss'])
            self.history['val_physics_loss'].append(val_metrics['physics_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Data: {train_metrics['data_loss']:.4f}, "
                  f"Physics: {train_metrics['physics_loss']:.4f})")
            if val_loader:
                print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                      f"(Data: {val_metrics['data_loss']:.4f}, "
                      f"Physics: {val_metrics['physics_loss']:.4f})")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_pinn.pth', epoch, val_metrics)
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model and history
        self.save_checkpoint('final_pinn.pth', epoch, val_metrics)
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history"""
        filepath = os.path.join(self.save_dir, 'pinn_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
