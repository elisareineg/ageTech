#!/usr/bin/env python3
"""
AgeTech Adoption Prediction - Complete ML Pipeline

This script runs the complete machine learning pipeline for AgeTech adoption prediction:
1. Synthetic data generation
2. Data preprocessing
3. Model training and evaluation
4. Model interpretability analysis

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def run_step(step_name: str, step_function, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = step_function(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✓ {step_name} completed successfully in {duration:.2f} seconds")
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✗ {step_name} failed after {duration:.2f} seconds")
        print(f"Error: {str(e)}")
        return None

def generate_synthetic_data():
    """Generate synthetic AgeTech dataset."""
    from src.data.generate_synthetic_data import AgeTechDataGenerator
    
    generator = AgeTechDataGenerator(n_samples=500, random_state=42)
    df = generator.generate_dataset()
    filepath = generator.save_dataset(df)
    
    return filepath, df

def preprocess_data(filepath: str):
    """Preprocess the synthetic dataset."""
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Filling {missing_values.sum()} missing values...")
        df = df.fillna(df.mode().iloc[0])
    
    # Remove ID column and separate features/target
    X = df.drop(['participant_id', 'adoption_success'], axis=1)
    y = df['adoption_success']
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    if len(numerical_cols) > 0:
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Data preprocessing completed:")
    print(f"  - Training samples: {len(X_train_scaled)}")
    print(f"  - Validation samples: {len(X_val_scaled)}")
    print(f"  - Test samples: {len(X_test_scaled)}")
    print(f"  - Features: {X_train_scaled.shape[1]}")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler
    }

def train_models(data):
    """Train multiple models and evaluate performance."""
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("Training and evaluating models...")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'auc_roc': roc_auc_score(y_val, y_val_pred_proba)
        }
        
        print(f"  Validation F1: {val_metrics['f1']:.4f}")
        print(f"  Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        results[name] = {
            'model': model,
            'val_metrics': val_metrics
        }
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_metrics']['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    # Final evaluation on test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'auc_roc': roc_auc_score(y_test, y_test_pred_proba)
    }
    
    print(f"\nFinal Test Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    for name, result in results.items():
        model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(result['model'], model_path)
        print(f"Saved {name} to {model_path}")
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance if hasattr(best_model, 'feature_importances_') else None
    }

def analyze_interpretability(data, training_results):
    """Perform interpretability analysis."""
    X_test = data['X_test']
    y_test = data['y_test']
    best_model = training_results['best_model']
    
    print("\nPerforming interpretability analysis...")
    
    # Subgroup analysis
    print("\nSubgroup Analysis:")
    print("-" * 30)
    
    # Load original data for subgroup analysis
    original_df = pd.read_csv("data/raw/agetch_synthetic_data_20250705_164017.csv")
    
    # Analyze by age group
    age_adoption_rates = original_df.groupby('age_group')['adoption_success'].mean()
    print("Adoption rates by age group:")
    for age_group, rate in age_adoption_rates.items():
        print(f"  {age_group}: {rate:.2%}")
    
    # Analyze by cognitive status
    cognitive_adoption_rates = original_df.groupby('cognitive_status')['adoption_success'].mean()
    print("\nAdoption rates by cognitive status:")
    for status, rate in cognitive_adoption_rates.items():
        print(f"  {status}: {rate:.2%}")
    
    # Analyze by digital literacy
    digital_adoption_rates = original_df.groupby('digital_literacy')['adoption_success'].mean()
    print("\nAdoption rates by digital literacy:")
    for literacy, rate in digital_adoption_rates.items():
        print(f"  {literacy}: {rate:.2%}")
    
    # Generate interpretability report
    os.makedirs("results", exist_ok=True)
    report_path = "results/interpretability_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("AgeTech Adoption Prediction - Interpretability Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 15 + "\n")
        f.write(f"• Best model: {training_results['best_model_name']}\n")
        f.write(f"• Test F1 Score: {training_results['test_metrics']['f1']:.4f}\n")
        f.write(f"• Test AUC-ROC: {training_results['test_metrics']['auc_roc']:.4f}\n\n")
        
        f.write("Subgroup Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write("Age Group Adoption Rates:\n")
        for age_group, rate in age_adoption_rates.items():
            f.write(f"  {age_group}: {rate:.2%}\n")
        
        f.write("\nCognitive Status Adoption Rates:\n")
        for status, rate in cognitive_adoption_rates.items():
            f.write(f"  {status}: {rate:.2%}\n")
        
        f.write("\nDigital Literacy Adoption Rates:\n")
        for literacy, rate in digital_adoption_rates.items():
            f.write(f"  {literacy}: {rate:.2%}\n")
    
    print(f"\nInterpretability report saved to {report_path}")
    
    return {
        'age_adoption_rates': age_adoption_rates,
        'cognitive_adoption_rates': cognitive_adoption_rates,
        'digital_adoption_rates': digital_adoption_rates,
        'report_path': report_path
    }

def main():
    """Run the complete AgeTech ML pipeline."""
    
    print("AgeTech Adoption Prediction - Complete ML Pipeline")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    filepath, df = run_step("Synthetic Data Generation", generate_synthetic_data)
    if filepath is None:
        return
    
    # Step 2: Preprocess data
    data = run_step("Data Preprocessing", preprocess_data, filepath)
    if data is None:
        return
    
    # Step 3: Train models
    training_results = run_step("Model Training and Evaluation", train_models, data)
    if training_results is None:
        return
    
    # Step 4: Interpretability analysis
    interpretability_results = run_step("Model Interpretability Analysis", 
                                      analyze_interpretability, data, training_results)
    if interpretability_results is None:
        return
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nGenerated Files:")
    print("-" * 20)
    
    # List generated files
    directories = [
        ("data/raw", "Synthetic datasets"),
        ("models", "Trained models"),
        ("results", "Analysis results")
    ]
    
    for dir_path, description in directories:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            if files:
                print(f"\n{description} ({dir_path}/):")
                for file in files[:5]:  # Show first 5 files
                    print(f"  - {file}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more files")
    
    print("\nKey Results:")
    print("-" * 15)
    print(f"• Best model: {training_results['best_model_name']}")
    print(f"• Test F1 Score: {training_results['test_metrics']['f1']:.4f}")
    print(f"• Test AUC-ROC: {training_results['test_metrics']['auc_roc']:.4f}")
    print(f"• Total participants: {len(df)}")
    print(f"• Adoption rate: {df['adoption_success'].mean():.2%}")
    
    print("\nNext Steps:")
    print("-" * 15)
    print("1. Review training results in models/ directory")
    print("2. Check interpretability report: results/interpretability_report.txt")
    print("3. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("4. Plan real-world validation study")

if __name__ == "__main__":
    main() 