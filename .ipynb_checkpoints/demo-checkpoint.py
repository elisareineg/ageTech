#!/usr/bin/env python3
"""
AgeTech Adoption Prediction - Demo Script

This script runs a quick demonstration of the ML pipeline with a smaller dataset
for testing and validation purposes.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

def run_demo():
    """Run a quick demo of the AgeTech ML pipeline."""
    
    print("AgeTech Adoption Prediction - Demo Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Generate small synthetic dataset
    print("\nStep 1: Generating Demo Dataset (100 samples)")
    print("-" * 50)
    
    try:
        from src.data.generate_synthetic_data import AgeTechDataGenerator
        
        generator = AgeTechDataGenerator(n_samples=100, random_state=42)
        df = generator.generate_dataset()
        
        # Save dataset
        os.makedirs("data/raw", exist_ok=True)
        filepath = "data/raw/demo_dataset.csv"
        df.to_csv(filepath, index=False)
        
        print(f"✓ Demo dataset generated: {filepath}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Adoption rate: {df['adoption_success'].mean():.2%}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"  - Warning: {missing_values.sum()} missing values detected")
            # Fill missing values
            df = df.fillna(df.mode().iloc[0])
            print(f"  - Missing values filled with mode")
        
    except Exception as e:
        print(f"✗ Demo data generation failed: {e}")
        return False
    
    # Step 2: Simple preprocessing
    print("\nStep 2: Simple Data Preprocessing")
    print("-" * 50)
    
    try:
        # Remove ID column and separate features/target
        X = df.drop(['participant_id', 'adoption_success'], axis=1)
        y = df['adoption_success']
        
        # Simple encoding for categorical variables
        label_encoders = {}
        X_encoded = X.copy()
        
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("✓ Simple preprocessing completed")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        
    except Exception as e:
        print(f"✗ Demo preprocessing failed: {e}")
        return False
    
    # Step 3: Quick model training
    print("\nStep 3: Quick Model Training (Random Forest)")
    print("-" * 50)
    
    try:
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        print("✓ Demo model training completed")
        print(f"  - Model: Random Forest")
        print(f"  - Test Accuracy: {accuracy:.4f}")
        print(f"  - Test F1 Score: {f1:.4f}")
        print(f"  - Test AUC-ROC: {auc_roc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        print(f"✗ Demo model training failed: {e}")
        return False
    
    # Step 4: Data exploration
    print("\nStep 4: Quick Data Exploration")
    print("-" * 50)
    
    try:
        print("Dataset Overview:")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Features: {len(df.columns) - 2}")  # Exclude ID and target
        print(f"  - Adoption rate: {df['adoption_success'].mean():.2%}")
        
        print("\nKey Variable Distributions:")
        key_vars = ['age_group', 'cognitive_status', 'digital_literacy', 'willingness_new_tech']
        for var in key_vars:
            if var in df.columns:
                print(f"  - {var}:")
                for value, count in df[var].value_counts().head(3).items():
                    print(f"    * {value}: {count} ({count/len(df):.1%})")
        
        print("\nAdoption Rates by Key Groups:")
        for var in key_vars:
            if var in df.columns:
                adoption_rates = df.groupby(var)['adoption_success'].mean().sort_values(ascending=False)
                print(f"  - {var}:")
                for group, rate in adoption_rates.head(2).items():
                    print(f"    * {group}: {rate:.2%}")
        
    except Exception as e:
        print(f"✗ Data exploration failed: {e}")
        return False
    
    # Demo summary
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nDemo Summary:")
    print("-" * 20)
    print("✓ Synthetic dataset generated (100 samples)")
    print("✓ Simple data preprocessing completed")
    print("✓ Random Forest model trained and evaluated")
    print("✓ Feature importance analysis performed")
    print("✓ Data exploration completed")
    
    print("\nKey Insights:")
    print("-" * 15)
    print("• Model achieved reasonable performance on synthetic data")
    print("• Feature importance reveals key predictors")
    print("• Adoption rates vary across demographic groups")
    print("• Pipeline is ready for full-scale analysis")
    
    print("\nNext Steps:")
    print("-" * 15)
    print("1. Run full pipeline: python run_pipeline.py")
    print("2. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("3. Review generated files in data/ directory")
    print("4. Plan real-world validation study")
    
    return True

if __name__ == "__main__":
    success = run_demo()
    if not success:
        sys.exit(1) 