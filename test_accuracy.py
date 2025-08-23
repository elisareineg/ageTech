#!/usr/bin/env python3
"""
AgeTech Adoption Prediction - Accuracy Testing
This script tests the actual accuracy of the ML model for AgeTech adoption prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🎯 AGETECH ADOPTION PREDICTION - ACCURACY TESTING")
    print("=" * 60)
    
    # Load the latest synthetic dataset
    try:
        df = pd.read_csv("data/raw/agetch_synthetic_data_20250823_164747.csv")
        print(f"✅ Dataset loaded! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run the pipeline first: python run_pipeline.py")
        return
    
    # Prepare the data
    print("\n📊 PREPARING DATA FOR MODEL TRAINING...")
    
    # Convert categorical variables to numerical
    df_encoded = df.copy()
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'participant_id':  # Skip ID column
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Select features and target
    feature_columns = [col for col in df_encoded.columns if col not in ['participant_id', 'adoption_success']]
    X = df_encoded[feature_columns]
    y = df_encoded['adoption_success']
    
    print(f"📈 Features: {len(feature_columns)}")
    print(f"🎯 Target: adoption_success")
    print(f"📊 Samples: {len(X)}")
    print(f"✅ Adoption rate: {y.mean():.2%}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n🔧 TRAINING MODELS...")
    print(f"📚 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n🏋️ Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        print(f"✅ {name} trained successfully!")
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 MODEL ACCURACY RESULTS")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n🎯 {name}:")
        print(f"   📈 Accuracy:     {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
        print(f"   🎯 Precision:    {metrics['precision']:.3f} ({metrics['precision']:.1%})")
        print(f"   🔄 Recall:       {metrics['recall']:.3f} ({metrics['recall']:.1%})")
        print(f"   ⚖️  F1-Score:     {metrics['f1']:.3f} ({metrics['f1']:.1%})")
        print(f"   📊 AUC-ROC:      {metrics['auc']:.3f} ({metrics['auc']:.1%})")
        print(f"   🔄 CV Accuracy:  {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   🎯 F1-Score: {best_model['f1']:.3f} ({best_model['f1']:.1%})")
    print(f"   📈 Accuracy:  {best_model['accuracy']:.3f} ({best_model['accuracy']:.1%})")
    
    # Feature importance for best model
    if hasattr(best_model['model'], 'feature_importances_'):
        print(f"\n🔍 TOP 10 FEATURE IMPORTANCE ({best_model_name}):")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:30s} | {row['importance']:.3f}")
    
    # Detailed classification report for best model
    best_model_instance = best_model['model']
    y_pred_best = best_model_instance.predict(X_test)
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, target_names=['No Adoption', 'Adoption']))
    
    # Summary
    print(f"\n🎉 ACCURACY TESTING COMPLETE!")
    print(f"📊 Best model achieves {best_model['accuracy']:.1%} accuracy")
    print(f"🎯 Best model achieves {best_model['f1']:.1%} F1-score")
    print(f"📈 Cross-validation shows {best_model['cv_mean']:.1%} ± {best_model['cv_std']:.1%} accuracy")
    
    # Check if accuracy meets the >90% claim
    if best_model['accuracy'] > 0.90:
        print(f"✅ EXCEEDS 90% ACCURACY TARGET!")
    else:
        print(f"⚠️  Below 90% accuracy target. Current best: {best_model['accuracy']:.1%}")

if __name__ == "__main__":
    main()
