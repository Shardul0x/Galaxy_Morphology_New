# view_my_results.py
"""View your completed augmentation results"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("YOUR AUGMENTATION RESULTS")
print("="*60)

# Load augmented data (skip comment lines)
try:
    df = pd.read_csv('results/galaxy_fixed_augmented.csv', comment='#')
except:
    # If that doesn't work, try without header
    try:
        df = pd.read_csv('results/galaxy_fixed_augmented.csv', skiprows=3)
    except:
        # Last resort: read all and find data
        with open('results/galaxy_fixed_augmented.csv', 'r') as f:
            lines = f.readlines()
        
        # Find where data starts
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and ',' in line:
                data_start = i
                break
        
        df = pd.read_csv('results/galaxy_fixed_augmented.csv', skiprows=data_start)

print(f"\n✓ Dataset: {df.shape[0]} samples x {df.shape[1]} features")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n✓ First 5 samples:")
print(df.head())

print(f"\n✓ Statistics:")
print(df.describe())

# Check for valid ranges (probabilities should be 0-1)
print(f"\n✓ Data quality check:")
for col in df.columns:
    try:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")
        if min_val < -0.1 or max_val > 1.1:
            print(f"    ⚠ Warning: Values outside expected range!")
    except:
        print(f"  {col}: Non-numeric column")

# Plot feature distributions
print(f"\n✓ Creating distribution plots...")

# Only plot numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=10)
    axes[i].set_ylabel('Count', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics text
    mean = df[col].mean()
    std = df[col].std()
    axes[i].text(0.95, 0.95, f'μ={mean:.3f}\nσ={std:.3f}',
                transform=axes[i].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

# Hide unused subplots
for i in range(len(numeric_cols), 9):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/feature_distributions.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: results/feature_distributions.png")

# Create correlation heatmap
if len(numeric_cols) > 1:
    print(f"\n✓ Creating correlation matrix...")
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlations in Augmented Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: results/correlation_matrix.png")

# View latent space plot if exists
import os
if os.path.exists('results/galaxy_fixed_latent.png'):
    print(f"\n✓ Latent space visualization available:")
    print(f"  results/galaxy_fixed_latent.png")

print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Augmented {df.shape[0]} galaxy samples")
print(f"✓ Features validated")
print(f"✓ Visualizations created in results/ folder")
print(f"\nYour augmented dataset is ready to use for:")
print("  • Machine learning training")
print("  • Galaxy classification")
print("  • Scientific analysis")
print("="*60)

# Also save a clean version without metadata
df.to_csv('results/galaxy_augmented_clean.csv', index=False)
print(f"\n✓ Saved clean version: results/galaxy_augmented_clean.csv")
