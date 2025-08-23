#!/usr/bin/env python3
"""
AgeTech Data Analysis Script
Runs the data exploration and shows results directly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔍 Loading your AgeTech dataset...")
    
    # Load data
    data_dir = Path('data/raw')
    csv_files = list(data_dir.glob('*.csv'))
    
    # Find latest file
    latest_file = None
    for file in csv_files:
        if 'agetch_synthetic_data' in file.name and 'demo' not in file.name:
            if latest_file is None or file.stat().st_mtime > latest_file.stat().st_mtime:
                latest_file = file
    
    if latest_file:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded: {latest_file.name}")
    else:
        df = pd.read_csv('data/raw/demo_dataset.csv')
        print("✅ Loaded demo dataset")
    
    print(f"📈 Shape: {df.shape}")
    print(f"🎯 Adoption rate: {df['adoption_success'].mean():.2%}")
    
    # Dataset overview
    print("\n📊 === DATASET OVERVIEW ===")
    print(f"👥 Total participants: {len(df)}")
    print(f"🔧 Features: {len(df.columns)}")
    print(f"🎯 Adoption success rate: {df['adoption_success'].mean():.2%}")
    
    print("\n📋 Data types:")
    print(df.dtypes.value_counts())
    
    print("\n🔍 Missing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ No missing values")
    
    # Key insights
    print("\n🎯 === KEY INSIGHTS ===")
    print(f"\n📊 1. Overall Adoption Rate: {df['adoption_success'].mean():.1%}")
    
    print("\n🏆 2. Highest Adoption Rates:")
    for col in ['age_group', 'cognitive_status', 'digital_literacy', 'willingness_new_tech']:
        if col in df.columns:
            adoption_by_group = df.groupby(col)['adoption_success'].mean().sort_values(ascending=False)
            top_group = adoption_by_group.index[0]
            top_rate = adoption_by_group.iloc[0]
            print(f"   - {col}: {top_group} ({top_rate:.1%})")
    
    print("\n🔍 3. Key Predictive Factors:")
    if 'technology_readiness_index' in df.columns:
        tech_corr = df['technology_readiness_index'].corr(df['adoption_success'])
        print(f"   - Technology Readiness Index: {tech_corr:.3f}")
    
    if 'health_risk_score' in df.columns:
        health_corr = df['health_risk_score'].corr(df['adoption_success'])
        print(f"   - Health Risk Score: {health_corr:.3f}")
    
    print("\n📋 4. Dataset Quality:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Features: {len(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    print("\n🎉 === ANALYSIS COMPLETE! ===")
    print("✅ You've successfully explored your AgeTech adoption data!")
    print("📊 All insights are ready for your research.")
    
    # Show some sample data
    print("\n📋 Sample Data (first 5 rows):")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main() 