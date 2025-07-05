"""
Model Training Module for AgeTech Adoption Prediction

This module implements multiple ML models with hyperparameter tuning:
- Gradient Boosting (XGBoost) - Primary model
- Random Forest
- Logistic Regression
- Cross-validation and hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AgeTechModelTrainer:
    """
    Comprehensive model training pipeline for AgeTech adoption prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def load_processed_data(self, data_dir: str = "data/processed") -> Dict:
        """Load processed training data."""
        
        try:
            X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
            X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
            X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            
            # Separate features and target
            y_train = X_train['adoption_success']
            y_val = X_val['adoption_success']
            y_test = X_test['adoption_success']
            
            X_train = X_train.drop('adoption_success', axis=1)
            X_val = X_val.drop('adoption_success', axis=1)
            X_test = X_test.drop('adoption_success', axis=1)
            
            data = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
            print(f"Data loaded successfully:")
            print(f"Train: {X_train.shape}")
            print(f"Validation: {X_val.shape}")
            print(f"Test: {X_test.shape}")
            
            return data
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return {}
    
    def define_models(self) -> Dict:
        """Define model configurations with hyperparameter grids."""
        
        models = {
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_samples': [10, 20, 30]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        return models
    
    def train_model_with_cv(self, model_name: str, model_config: Dict, 
                           X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict]:
        """Train a single model with cross-validation and hyperparameter tuning."""
        
        print(f"\nTraining {model_name}...")
        print("-" * 40)
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=cv, scoring='f1'
        )
        
        # Store results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"CV mean ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_model, results
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      dataset_name: str = "Validation") -> Dict:
        """Evaluate model performance on a dataset."""
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }
        
        print(f"\n{dataset_name} Set Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str], 
                              model_name: str) -> pd.DataFrame:
        """Extract feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[model_name] = importance_df
        
        return importance_df
    
    def train_all_models(self, data: Dict) -> Dict:
        """Train all models and evaluate performance."""
        
        print("Starting model training pipeline...")
        print("=" * 60)
        
        # Extract data
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Get feature names
        feature_names = X_train.columns.tolist()
        
        # Define models
        model_configs = self.define_models()
        
        # Training results
        training_results = {}
        
        # Train each model
        for model_name, model_config in model_configs.items():
            try:
                # Train model with CV
                best_model, cv_results = self.train_model_with_cv(
                    model_name, model_config, X_train, y_train
                )
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model(best_model, X_val, y_val, "Validation")
                
                # Get feature importance
                importance_df = self.get_feature_importance(
                    best_model, feature_names, model_name
                )
                
                # Store results
                training_results[model_name] = {
                    'model': best_model,
                    'cv_results': cv_results,
                    'val_metrics': val_metrics,
                    'feature_importance': importance_df
                }
                
                self.best_models[model_name] = best_model
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        return training_results
    
    def select_best_model(self, training_results: Dict) -> Tuple[str, Any]:
        """Select the best model based on validation F1 score."""
        
        best_model_name = None
        best_f1_score = 0
        
        print("\nModel Comparison:")
        print("=" * 50)
        print(f"{'Model':<20} {'F1 Score':<10} {'AUC-ROC':<10}")
        print("-" * 50)
        
        for model_name, results in training_results.items():
            f1_score = results['val_metrics']['f1']
            auc_roc = results['val_metrics']['auc_roc']
            
            print(f"{model_name:<20} {f1_score:<10.4f} {auc_roc:<10.4f}")
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_name = model_name
        
        print("-" * 50)
        print(f"Best model: {best_model_name} (F1: {best_f1_score:.4f})")
        
        return best_model_name, self.best_models[best_model_name]
    
    def final_evaluation(self, best_model: Any, data: Dict) -> Dict:
        """Final evaluation on test set."""
        
        print(f"\nFinal Evaluation on Test Set:")
        print("=" * 40)
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(best_model, X_test, y_test, "Test")
        
        # Detailed classification report
        y_pred = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return test_metrics
    
    def save_models(self, training_results: Dict, best_model_name: str, 
                   output_dir: str = "models"):
        """Save trained models and results."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        best_model_path = os.path.join(output_dir, f"best_model_{best_model_name}.pkl")
        joblib.dump(self.best_models[best_model_name], best_model_path)
        print(f"Best model saved to {best_model_path}")
        
        # Save all models
        for model_name, model in self.best_models.items():
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
        
        # Save training results
        results_path = os.path.join(output_dir, "training_results.pkl")
        joblib.dump(training_results, results_path)
        print(f"Training results saved to {results_path}")
        
        # Save feature importance
        importance_path = os.path.join(output_dir, "feature_importance.pkl")
        joblib.dump(self.feature_importance, importance_path)
        print(f"Feature importance saved to {importance_path}")
    
    def generate_training_report(self, training_results: Dict, best_model_name: str, 
                                test_metrics: Dict) -> str:
        """Generate a comprehensive training report."""
        
        report = []
        report.append("AgeTech Adoption Prediction - Model Training Report")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison
        report.append("Model Performance Comparison:")
        report.append("-" * 40)
        report.append(f"{'Model':<20} {'F1 Score':<10} {'AUC-ROC':<10} {'Precision':<10} {'Recall':<10}")
        report.append("-" * 70)
        
        for model_name, results in training_results.items():
            metrics = results['val_metrics']
            report.append(
                f"{model_name:<20} {metrics['f1']:<10.4f} {metrics['auc_roc']:<10.4f} "
                f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}"
            )
        
        report.append("")
        report.append(f"Best Model: {best_model_name}")
        report.append("")
        
        # Test set performance
        report.append("Final Test Set Performance:")
        report.append("-" * 30)
        for metric, value in test_metrics.items():
            report.append(f"{metric.capitalize()}: {value:.4f}")
        
        report.append("")
        
        # Feature importance for best model
        if best_model_name in self.feature_importance:
            report.append("Top 10 Most Important Features:")
            report.append("-" * 30)
            top_features = self.feature_importance[best_model_name].head(10)
            for _, row in top_features.iterrows():
                report.append(f"{row['feature']}: {row['importance']:.4f}")
        
        return "\n".join(report)
    
    def run_training_pipeline(self) -> Dict:
        """Complete training pipeline."""
        
        print("AgeTech Model Training Pipeline")
        print("=" * 50)
        
        # Load data
        data = self.load_processed_data()
        if not data:
            print("Failed to load processed data!")
            return {}
        
        # Train all models
        training_results = self.train_all_models(data)
        
        if not training_results:
            print("No models were successfully trained!")
            return {}
        
        # Select best model
        best_model_name, best_model = self.select_best_model(training_results)
        
        # Final evaluation
        test_metrics = self.final_evaluation(best_model, data)
        
        # Save models and results
        self.save_models(training_results, best_model_name)
        
        # Generate and save report
        report = self.generate_training_report(training_results, best_model_name, test_metrics)
        
        report_path = "results/training_report.txt"
        os.makedirs("results", exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nTraining report saved to {report_path}")
        print("\nTraining pipeline completed successfully!")
        
        return {
            'training_results': training_results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'test_metrics': test_metrics,
            'report': report
        }

def main():
    """Run the complete training pipeline."""
    
    # Initialize trainer
    trainer = AgeTechModelTrainer(random_state=42)
    
    # Run training pipeline
    results = trainer.run_training_pipeline()
    
    if results:
        print("\nTraining Summary:")
        print("=" * 30)
        print(f"Best model: {results['best_model_name']}")
        print(f"Test F1 score: {results['test_metrics']['f1']:.4f}")
        print(f"Test AUC-ROC: {results['test_metrics']['auc_roc']:.4f}")

if __name__ == "__main__":
    main() 