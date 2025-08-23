#!/usr/bin/env python3
"""
Better Correlation Matrix for AgeTech Data
Run this script to see a much more readable correlation matrix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/raw/agetch_synthetic_data_20250823_145149.csv")

# Convert categorical to numerical
df_numeric = df.copy()
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    df_numeric[col] = pd.Categorical(df_numeric[col]).codes

# Get top correlations
adoption_correlations = df_numeric.corr()["adoption_success"].abs().sort_values(ascending=False)
top_correlations = adoption_correlations.head(11)[1:]  # Exclude adoption_success itself

print("🔍 TOP 10 CORRELATIONS WITH ADOPTION SUCCESS:")
print("=" * 50)
for i, (feature, corr) in enumerate(top_correlations.items(), 1):
    print(f"{i:2d}. {feature:30s} | {corr:.3f}")

# Create focused correlation matrix (only top 8 predictors)
important_features = top_correlations.head(8).index.tolist() + ["adoption_success"]
focused_corr = df_numeric[important_features].corr()

# Plot improved correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(focused_corr, dtype=bool))
sns.heatmap(focused_corr, mask=mask, annot=True, cmap="RdYlBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt=".2f",
            annot_kws={"size": 12})
plt.title("Correlation Matrix - Top 8 Predictors + Adoption Success", fontsize=16, pad=20)
plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

print("\n✅ This correlation matrix is much easier to read!")
print("• Shows only the most important 8 predictors")
print("• Larger text and better colors")
print("• Focuses on the relationships that matter most")
