# visualization/plotter.py
"""
Scientific Visualization Tools for Astrophysics Data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class AstrophysicsVisualizer:
    """
    Visualization tools for astrophysics data augmentation
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_latent_space_2d(
        self,
        latent_vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Latent Space (2D Projection)",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot 2D projection of latent space
        
        Args:
            latent_vectors: Latent representations [n_samples, latent_dim]
            labels: Optional labels for coloring
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D if needed
        if latent_vectors.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_vectors)
            explained_var = pca.explained_variance_ratio_
        else:
            latent_2d = latent_vectors
            explained_var = [1.0, 0.0]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                c=labels, cmap='viridis', alpha=0.6, s=20
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1],
                alpha=0.6, s=20, color='steelblue'
            )
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_latent_space_3d(
        self,
        latent_vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Latent Space (3D Projection)",
        save_path: Optional[str] = None
    ) -> Figure:
        """Plot 3D projection of latent space"""
        from sklearn.decomposition import PCA
        
        # Reduce to 3D if needed
        if latent_vectors.shape[1] > 3:
            pca = PCA(n_components=3)
            latent_3d = pca.fit_transform(latent_vectors)
        else:
            latent_3d = latent_vectors[:, :3]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                c=labels, cmap='viridis', alpha=0.6, s=20
            )
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            ax.scatter(
                latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                alpha=0.6, s=20, color='steelblue'
            )
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_reconstruction_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_idx: int = 0,
        title: str = "Original vs Reconstructed",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Compare original and reconstructed features
        
        Args:
            original: Original features [n_samples, n_features]
            reconstructed: Reconstructed features [n_samples, n_features]
            feature_names: Names of features
            sample_idx: Index of sample to plot
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(original.shape[1])
        
        ax.plot(x, original[sample_idx], 'o-', label='Original', linewidth=2, markersize=8)
        ax.plot(x, reconstructed[sample_idx], 's--', label='Reconstructed', linewidth=2, markersize=8)
        
        if feature_names:
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
        else:
            ax.set_xlabel('Feature Index')
        
        ax.set_ylabel('Feature Value')
        ax.set_title(f'{title} (Sample {sample_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_evolution(
        self,
        evolution_data: np.ndarray,
        time_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_indices: Optional[List[int]] = None,
        title: str = "Temporal Evolution",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot temporal evolution of features
        
        Args:
            evolution_data: Evolution data [n_timesteps, n_features]
            time_values: Optional time values
            feature_names: Names of features
            feature_indices: Indices of features to plot (default: all)
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            Matplotlib figure
        """
        if time_values is None:
            time_values = np.arange(len(evolution_data))
        
        if feature_indices is None:
            feature_indices = list(range(evolution_data.shape[1]))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx in feature_indices:
            label = feature_names[idx] if feature_names else f'Feature {idx}'
            ax.plot(time_values, evolution_data[:, idx], marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Feature Value')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict,
        metrics: List[str] = ['loss'],
        title: str = "Training History",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics // 2))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                ax.plot(history[train_key], label=f'Train {metric}', linewidth=2)
            
            if val_key in history:
                ax.plot(history[val_key], label=f'Val {metric}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_cols: int = 3,
        title: str = "Feature Distributions",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot distributions of all features
        
        Args:
            data: Feature data [n_samples, n_features]
            feature_names: Names of features
            n_cols: Number of columns in subplot grid
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            Matplotlib figure
        """
        n_features = data.shape[1]
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i in range(n_features):
            ax = axes[i]
            ax.hist(data[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            
            label = feature_names[i] if feature_names else f'Feature {i}'
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_evolution_animation(
        self,
        evolution_sequence: List[np.ndarray],
        time_values: np.ndarray,
        feature_idx: int = 0,
        feature_name: Optional[str] = None,
        save_path: Optional[str] = None,
        fps: int = 10
    ) -> animation.FuncAnimation:
        """
        Create animation of temporal evolution
        
        Args:
            evolution_sequence: List of evolution states
            time_values: Time values for each state
            feature_idx: Feature index to animate
            feature_name: Name of feature
            save_path: Optional path to save animation (requires ffmpeg)
            fps: Frames per second
            
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        line, = ax.plot([], [], 'o-', linewidth=2, markersize=8)
        
        feature_label = feature_name if feature_name else f'Feature {feature_idx}'
        ax.set_xlabel('Time')
        ax.set_ylabel(feature_label)
        ax.set_title(f'Evolution of {feature_label}')
        ax.grid(True, alpha=0.3)
        
        # Set limits
        all_values = np.concatenate([state[:, feature_idx] for state in evolution_sequence])
        ax.set_xlim(time_values[0], time_values[-1])
        ax.set_ylim(all_values.min() * 0.9, all_values.max() * 1.1)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            x = time_values[:i+1]
            y = evolution_sequence[0][:i+1, feature_idx]
            line.set_data(x, y)
            return line,
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(time_values), interval=1000/fps,
            blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        
        return anim
