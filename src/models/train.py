"""
Model Training for AgeTech

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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, skipping XGBoost models")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available, skipping LightGBM models")

import joblib
import os
import gc  # Garbage collection for memory optimization
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
        """Load processed training data with memory optimization."""
        
        try:
            print("Loading data with memory optimization...")
            
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
            
            print(f"Loaded data shapes: Train{X_train.shape}, Val{X_val.shape}, Test{X_test.shape}")
            
            # Smart feature selection to reduce memory usage
            X_train_selected, X_val_selected, X_test_selected, selected_features = self.smart_feature_selection(
                X_train, X_val, X_test, y_train
            )
            
            print(f"Selected {len(selected_features)} features: {selected_features[:10]}..." if len(selected_features) > 10 else selected_features)
            
            data = {
                'X_train': X_train_selected,
                'X_val': X_val_selected,
                'X_test': X_test_selected,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
            print(f"Data loaded successfully:")
            print(f"Train: {X_train_selected.shape}")
            print(f"Validation: {X_val_selected.shape}")
            print(f"Test: {X_test_selected.shape}")
            
            return data
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return {}
    
    def smart_feature_selection(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                               X_test: pd.DataFrame, y_train: pd.Series, 
                               max_features: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Smart feature selection combining statistical tests and model-based selection.
        """
        
        print(f"Performing smart feature selection from {X_train.shape[1]} features...")
        
        # Step 1: Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=0.01)
        X_train_var = var_selector.fit_transform(X_train)
        var_features = X_train.columns[var_selector.get_support()].tolist()
        
        # Step 2: Statistical feature selection
        from sklearn.feature_selection import SelectKBest, f_classif
        stat_selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, len(var_features)))
        X_train_stat = stat_selector.fit_transform(X_train[var_features], y_train)
        stat_features = [var_features[i] for i in stat_selector.get_support(indices=True)]
        
        # Step 3: Model-based feature selection using a fast model
        print("Performing model-based feature selection...")
        from sklearn.feature_selection import RFE
        fast_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=self.random_state, n_jobs=1)
        model_selector = RFE(estimator=fast_model, n_features_to_select=max_features, step=1)
        
        X_train_model = model_selector.fit_transform(X_train[stat_features], y_train)
        selected_features = [stat_features[i] for i in model_selector.get_support(indices=True)]
        
        # Apply selection to all sets
        X_train_selected = pd.DataFrame(X_train_model, columns=selected_features, index=X_train.index)
        X_val_selected = pd.DataFrame(model_selector.transform(X_val[stat_features]), columns=selected_features, index=X_val.index)
        X_test_selected = pd.DataFrame(model_selector.transform(X_test[stat_features]), columns=selected_features, index=X_test.index)
        
        # Save selectors
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_selector, "models/feature_selector.pkl")
        
        # Force garbage collection
        gc.collect()
        
        return X_train_selected, X_val_selected, X_test_selected, selected_features
    
    def define_models(self) -> Dict:
        """Define model configurations with memory-optimized hyperparameter grids."""
        
        models = {}
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=self.random_state, n_jobs=1),  # Single thread
                'params': {
                    'n_estimators': [100, 200],  # Reduced options
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, n_jobs=1),  # Single thread
                'params': {
                    'n_estimators': [100, 200],  # Reduced options
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_samples': [10, 20, 30]
                }
            }
        
        # Memory-optimized Random Forest
        models['random_forest'] = {
            'model': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=1,  # Single thread to prevent memory issues
                class_weight='balanced'
            ),
            'params': {
                'n_estimators': [100, 200, 300],  # Reduced from 500
                'max_depth': [5, 10, 15],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        }
        
        # Memory-optimized Gradient Boosting
        models['gradient_boosting'] = {
            'model': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Logistic Regression (memory efficient)
        models['logistic_regression'] = {
            'model': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],  # Only L2 to reduce complexity
                'solver': ['liblinear']
            }
        }
        
        return models
    
    def train_model_with_cv(self, model_name: str, model_config: Dict, 
                           X_train: pd.DataFrame, y_train: pd.Series, data: Dict) -> Tuple[Any, Dict]:
        """Train a single model with memory-efficient cross-validation and hyperparameter tuning."""
        
        print(f"\nTraining {model_name}...")
        print("-" * 40)
        
        # Use smaller CV to prevent memory issues
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # Reduced from 5
        
        # Bayesian optimization for hyperparameter tuning
        print("Using Bayesian optimization for hyperparameter tuning...")
        
        try:
            from skopt import BayesSearchCV
            bayes_search = BayesSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring='f1',
                n_iter=10,  # Reduced iterations for memory efficiency
                random_state=self.random_state,
                n_jobs=1  # Single thread
            )
            
            bayes_search.fit(X_train, y_train)
            best_model = bayes_search.best_estimator_
            best_params = bayes_search.best_params_
            cv_scores = bayes_search.cv_results_
            
            print(f"Best parameters: {best_params}")
            print(f"CV mean ± std: {bayes_search.best_score_:.4f} ± {bayes_search.cv_results_['std_test_score'][bayes_search.best_index_]:.4f}")
            
        except ImportError:
            print("scikit-optimize not available, using default parameters...")
            # Use simple parameter search instead of extensive grid search
            best_model = model_config['model']
            best_params = best_model.get_params()
            
            # Try a few parameter combinations manually
            if model_name == 'random_forest':
                param_combinations = [
                    {'n_estimators': 100, 'max_depth': 8},
                    {'n_estimators': 200, 'max_depth': 10},
                    {'n_estimators': 150, 'max_depth': 12}
                ]
            elif model_name == 'gradient_boosting':
                param_combinations = [
                    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                    {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6},
                    {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5}
                ]
            else:
                param_combinations = [
                    {'C': 1.0},
                    {'C': 10.0},
                    {'C': 0.1}
                ]
            
            best_score = 0
            for params in param_combinations:
                try:
                    # Create model with current parameters
                    if model_name == 'random_forest':
                        current_model = RandomForestClassifier(
                            random_state=self.random_state, n_jobs=1, **params
                        )
                    elif model_name == 'gradient_boosting':
                        current_model = GradientBoostingClassifier(
                            random_state=self.random_state, **params
                        )
                    else:
                        current_model = LogisticRegression(
                            random_state=self.random_state, max_iter=1000, 
                            solver='liblinear', **params
                        )
                    
                    # Cross-validation
                    cv_scores = cross_val_score(current_model, X_train, y_train, cv=cv, scoring='f1')
                    mean_score = cv_scores.mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = current_model
                        best_params = params
                    
                    print(f"  {params}: CV F1 = {mean_score:.4f}")
                    
                    # Force garbage collection after each model
                    gc.collect()
                    
                except Exception as e:
                    print(f"  Error with {params}: {e}")
                    continue
            
            print(f"Best parameters: {best_params}")
            print(f"CV mean ± std: {best_score:.4f} ± {cv_scores.std():.4f}")
        
        # Train best model on full training set
        best_model.fit(X_train, y_train)
        
        # Get cross-validation scores for best model
        final_cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1')
        
        results = {
            'best_params': best_params,
            'cv_mean': final_cv_scores.mean(),
            'cv_std': final_cv_scores.std(),
            'cv_scores': final_cv_scores
        }
        
        print(f"Best CV score: {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}")
        
        # Force garbage collection
        gc.collect()
        
        return best_model, results
    
    def evaluate_model_safe(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Safe model evaluation with memory management."""
        
        try:
            # Make predictions in batches if dataset is large
            if len(X) > 5000:
                batch_size = 1000
                y_pred_list = []
                y_pred_proba_list = []
                
                for i in range(0, len(X), batch_size):
                    batch_X = X.iloc[i:i+batch_size]
                    y_pred_batch = model.predict(batch_X)
                    y_pred_proba_batch = model.predict_proba(batch_X)[:, 1]
                    
                    y_pred_list.append(y_pred_batch)
                    y_pred_proba_list.append(y_pred_proba_batch)
                
                y_pred = np.concatenate(y_pred_list)
                y_pred_proba = np.concatenate(y_pred_proba_list)
            else:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y, y_pred_proba)
            }
            
            # Force garbage collection
            gc.collect()
            
            return metrics
        
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1': 0.0, 'auc_roc': 0.5
            }
    
    def get_feature_importance_safe(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Safely extract feature importance."""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return pd.DataFrame()
    
    def run_training_pipeline(self) -> Dict:
        """Run the complete training pipeline with memory optimization."""
        
        print("AgeTech Model Training Pipeline")
        print("=" * 50)
        
        # Load processed data
        data = self.load_processed_data()
        if not data:
            print("Failed to load data")
            return {}
        
        # Define models
        model_configs = self.define_models()
        training_results = {}
        
        # Train models sequentially to manage memory
        for model_name, model_config in model_configs.items():
            try:
                print(f"\n{'='*20} {model_name.upper()} {'='*20}")
                
                # Train model
                best_model, cv_results = self.train_model_with_cv(
                    model_name, model_config, data['X_train'], data['y_train'], data
                )
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model_safe(
                    best_model, data['X_val'], data['y_val']
                )
                
                # Get feature importance
                importance_df = self.get_feature_importance_safe(
                    best_model, data['X_train'].columns.tolist()
                )
                
                # Store results
                training_results[model_name] = {
                    'model': best_model,
                    'cv_results': cv_results,
                    'val_metrics': val_metrics,
                    'feature_importance': importance_df
                }
                
                self.best_models[model_name] = best_model
                self.feature_importance[model_name] = importance_df
                
                print(f"✓ {model_name} completed - Val F1: {val_metrics['f1']:.4f}")
                
                # Force garbage collection after each model
                gc.collect()
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                continue
        
        # Select best model
        best_model_name = self.select_best_model(training_results)
        
        if best_model_name:
            # Final evaluation on test set
            test_metrics = self.evaluate_model_safe(
                self.best_models[best_model_name], 
                data['X_test'], 
                data['y_test']
            )
            
            # Save results
            self.save_models(training_results, best_model_name)
            
            # Generate report
            report = self.generate_training_report(
                training_results, best_model_name, test_metrics
            )
            
            return {
                'training_results': training_results,
                'best_model_name': best_model_name,
                'test_metrics': test_metrics,
                'report': report
            }
        
        return training_results
    
    def select_best_model(self, training_results: Dict) -> str:
        """Select the best model based on validation F1 score."""
        
        if not training_results:
            print("No training results available!")
            return None
        
        best_model_name = None
        best_f1_score = 0
        
        print("\nModel Comparison:")
        print("-" * 50)
        print(f"{'Model':<20} {'Val F1':<10} {'Val AUC':<10}")
        print("-" * 50)
        
        for model_name, results in training_results.items():
            if 'val_metrics' in results:
                f1_score = results['val_metrics']['f1']
                auc_score = results['val_metrics']['auc_roc']
                
                print(f"{model_name:<20} {f1_score:<10.4f} {auc_score:<10.4f}")
                
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_model_name = model_name
        
        print("-" * 50)
        if best_model_name:
            print(f"Best model: {best_model_name} (F1: {best_f1_score:.4f})")
        else:
            print("No valid model found!")
        
        return best_model_name
    
    def save_models(self, training_results: Dict, best_model_name: str):
        """Save models and results with memory optimization."""
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        try:
            # Save best model
            if best_model_name and best_model_name in self.best_models:
                best_model_path = f"models/best_model_{best_model_name}.pkl"
                joblib.dump(self.best_models[best_model_name], best_model_path)
                print(f"✓ Best model saved: {best_model_path}")
            
            # Save feature importance
            if self.feature_importance:
                importance_path = "models/feature_importance.pkl"
                joblib.dump(self.feature_importance, importance_path)
                print(f"✓ Feature importance saved: {importance_path}")
            
            # Save training results (without models to save space)
            results_to_save = {}
            for name, results in training_results.items():
                results_to_save[name] = {
                    'cv_results': results.get('cv_results', {}),
                    'val_metrics': results.get('val_metrics', {}),
                    'feature_importance': results.get('feature_importance', pd.DataFrame())
                }
            
            results_path = "models/training_results.pkl"
            joblib.dump(results_to_save, results_path)
            print(f"✓ Training results saved: {results_path}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def generate_training_report(self, training_results: Dict, 
                                best_model_name: str, test_metrics: Dict) -> str:
        """Generate a comprehensive training report."""
        
        report = []
        report.append("AgeTech Model Training Report - Memory Optimized")
        report.append("=" * 60)
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for model_name, results in training_results.items():
            if 'val_metrics' in results:
                metrics = results['val_metrics']
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  Validation F1:    {metrics.get('f1', 0):.4f}")
                report.append(f"  Validation AUC:   {metrics.get('auc_roc', 0):.4f}")
                report.append(f"  Validation Acc:   {metrics.get('accuracy', 0):.4f}")
        
        report.append("")
        report.append(f"BEST MODEL: {best_model_name}")
        report.append("-" * 20)
        
        if test_metrics:
            report.append("Test Set Performance:")
            report.append(f"  Accuracy:   {test_metrics.get('accuracy', 0):.1%}")
            report.append(f"  Precision:  {test_metrics.get('precision', 0):.1%}")
            report.append(f"  Recall:     {test_metrics.get('recall', 0):.1%}")
            report.append(f"  F1 Score:   {test_metrics.get('f1', 0):.1%}")
            report.append(f"  AUC-ROC:    {test_metrics.get('auc_roc', 0):.4f}")
        
        # Top features
        if best_model_name in self.feature_importance:
            importance_df = self.feature_importance[best_model_name]
            if not importance_df.empty:
                report.append("")
                report.append("TOP 10 FEATURES:")
                report.append("-" * 20)
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    report.append(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        report.append("")
        report.append("MEMORY OPTIMIZATION NOTES:")
        report.append("-" * 30)
        report.append("• Used reduced cross-validation (3-fold)")
        report.append("• Applied smart feature selection")
        report.append("• Implemented batch processing for large datasets")
        report.append("• Used single-threaded processing to prevent crashes")
        report.append("• Applied aggressive garbage collection")
        
        return "\n".join(report)

def main():
    """Run the training pipeline."""
    
    try:
        # Initialize trainer
        trainer = AgeTechModelTrainer(random_state=42)
        
        # Run training
        results = trainer.run_training_pipeline()
        
        if results and 'best_model_name' in results:
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Best Model: {results['best_model_name']}")
            
            if 'test_metrics' in results:
                test_metrics = results['test_metrics']
                print(f"Test F1 Score: {test_metrics.get('f1', 0):.1%}")
                print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.1%}")
                print(f"Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
            
            # Save report
            if 'report' in results:
                report_path = "results/training_report.txt"
                with open(report_path, 'w') as f:
                    f.write(results['report'])
                print(f"\nReport saved: {report_path}")
        else:
            print("Training completed with issues. Check logs above.")
            
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 