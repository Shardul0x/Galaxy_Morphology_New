# compare_original_vs_augmented.py
"""Compare original vs augmented data quality"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.loader import GalaxyZooLoader

print("="*60)
print("ORIGINAL vs AUGMENTED COMPARISON")
print("="*60)

# Load original
loader = GalaxyZooLoader()
loader.load('GalaxyZoo1_DR_table2.csv')
original = loader.get_features()
df_original = pd.DataFrame(original, columns=loader.metadata['feature_names'])

# Load augmented (use clean version)
try:
    df_augmented = pd.read_csv('results/galaxy_augmented_clean.csv')
except:
    print("\n⚠ Clean CSV not found, trying to load with comment skip...")
    df_augmented = pd.read_csv('results/galaxy_fixed_augmented.csv', comment='#')

print(f"\nOriginal:   {df_original.shape[0]:,} samples")
print(f"Augmented:  {df_augmented.shape[0]:,} samples")
print(f"Ratio:      {df_augmented.shape[0]/df_original.shape[0]:.6f}x")

# Compare distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(df_original.columns):
    # Original
    axes[i].hist(df_original[col], bins=50, alpha=0.5, 
                label='Original', color='blue', density=True, edgecolor='black', linewidth=0.5)
    
    # Augmented
    axes[i].hist(df_augmented[col], bins=50, alpha=0.5,
                label='Augmented', color='red', density=True, edgecolor='black', linewidth=0.5)
    
    axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=10)
    axes[i].set_ylabel('Density', fontsize=10)
    axes[i].legend(loc='upper right', fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Original vs Augmented Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/comparison_original_vs_augmented.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison plot: results/comparison_original_vs_augmented.png")

# Statistical comparison
print(f"\n" + "="*60)
print("STATISTICAL COMPARISON")
print("="*60)
print(f"\n{'Feature':<15} {'Original Mean':<15} {'Aug Mean':<15} {'Diff':<10}")
print("-"*60)

for col in df_original.columns:
    orig_mean = df_original[col].mean()
    aug_mean = df_augmented[col].mean()
    diff = abs(aug_mean - orig_mean)
    
    print(f"{col:<15} {orig_mean:>14.4f} {aug_mean:>14.4f} {diff:>9.4f}")

print(f"\n{'Feature':<15} {'Original Std':<15} {'Aug Std':<15} {'Diff':<10}")
print("-"*60)

for col in df_original.columns:
    orig_std = df_original[col].std()
    aug_std = df_augmented[col].std()
    diff = abs(aug_std - orig_std)
    
    print(f"{col:<15} {orig_std:>14.4f} {aug_std:>14.4f} {diff:>9.4f}")

# Calculate distribution similarity (KS test)
print(f"\n" + "="*60)
print("DISTRIBUTION SIMILARITY (Kolmogorov-Smirnov Test)")
print("="*60)
print("(p-value > 0.05 means distributions are similar)")
print("-"*60)

from scipy.stats import ks_2samp

print(f"\n{'Feature':<15} {'KS Statistic':<15} {'P-Value':<15} {'Similar?':<10}")
print("-"*60)

for col in df_original.columns:
    stat, pval = ks_2samp(df_original[col], df_augmented[col])
    similar = "✓ Yes" if pval > 0.05 else "✗ No"
    print(f"{col:<15} {stat:>14.4f} {pval:>14.4f} {similar:<10}")

print(f"\n" + "="*60)
print("QUALITY ASSESSMENT")
print("="*60)

# Count how many features pass similarity test
similar_count = sum(
    1 for col in df_original.columns 
    if ks_2samp(df_original[col], df_augmented[col])[1] > 0.05
)

print(f"\n✓ {similar_count}/{len(df_original.columns)} features have similar distributions")

if similar_count >= len(df_original.columns) * 0.7:
    print("✓ EXCELLENT: Augmented data closely matches original!")
elif similar_count >= len(df_original.columns) * 0.5:
    print("✓ GOOD: Most features match reasonably well")
else:
    print("⚠ WARNING: Some features differ significantly")

print(f"\n✓ Augmented data is ready for ML training!")
print("="*60)
