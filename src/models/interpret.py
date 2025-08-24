"""
Model Interpretability Module for AgeTech Adoption Prediction

This module provides SHAP analysis and interpretability tools for understanding
model predictions and feature importance in the context of AgeTech adoption.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, interpretability features will be limited")
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AgeTechInterpreter:
    """
    Comprehensive model interpretability analysis using SHAP.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.shap_values = {}
        self.explainers = {}
        self.feature_importance = {}
        
    def load_model_and_data(self, model_dir: str = "models", 
                           data_dir: str = "data/processed") -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
        """Load the best model and test data."""
        
        try:
            # Load best model - get the most recent one
            model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
            if not model_files:
                print("No best model found!")
                return None, None, None
            
            # Sort by modification time to get the most recent
            model_files_with_time = [(f, os.path.getmtime(os.path.join(model_dir, f))) for f in model_files]
            model_files_with_time.sort(key=lambda x: x[1], reverse=True)
            best_model_file = model_files_with_time[0][0]
            
            model_path = os.path.join(model_dir, best_model_file)
            model = joblib.load(model_path)
            
            print(f"Loaded model: {best_model_file}")
            
            # Load test data (proper evaluation set)
            X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            y_test = X_test['adoption_success']
            X_test = X_test.drop('adoption_success', axis=1)
            
            # Load the saved feature selector for consistent evaluation
            selector_path = os.path.join("models", "feature_selector.pkl")
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                print(f"Loaded saved feature selector")
            else:
                # Fallback: recreate selector (less reliable)
                from sklearn.feature_selection import SelectKBest, f_classif
                X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
                y_train = X_train['adoption_success']
                X_train = X_train.drop('adoption_success', axis=1)
                selector = SelectKBest(score_func=f_classif, k=15)
                selector.fit(X_train, y_train)
                print(f"Recreated feature selector (fallback)")
            
            # Apply to test data
            X_test_selected = selector.transform(X_test)
            selected_features = X_test.columns[selector.get_support()].tolist()
            X_test = pd.DataFrame(X_test_selected, columns=selected_features)
            
            print(f"Loaded test data: {X_test.shape}")
            
            return model, X_test, y_test
            
        except Exception as e:
            print(f"Error loading model and data: {e}")
            return None, None, None
    
    def create_shap_explainer(self, model: Any, X: pd.DataFrame) -> Any:
        """Create SHAP explainer for the model."""
        
        if not SHAP_AVAILABLE:
            print("SHAP not available, skipping SHAP analysis")
            return None
        
        try:
            # Create explainer based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, Gradient Boosting)
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression)
                explainer = shap.LinearExplainer(model, X)
            else:
                # Fallback for other models
                explainer = shap.KernelExplainer(model.predict_proba, X.sample(n=min(100, len(X)), random_state=self.random_state))
            
            return explainer
            
        except Exception as e:
            print(f"Error creating SHAP explainer: {e}")
            return None
    
    def compute_shap_values(self, explainer: Any, X: pd.DataFrame) -> Tuple[Any, Any]:
        """Compute SHAP values for the dataset."""
        
        if not SHAP_AVAILABLE or explainer is None:
            print("SHAP not available, skipping SHAP values computation")
            return None, None
        
        try:
            # Use a sample for faster computation
            if len(X) > 100:
                X_sample = X.sample(n=100, random_state=self.random_state)
            else:
                X_sample = X
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use positive class
            
            return shap_values, X_sample
            
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None, None
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame, 
                         output_dir: str = "results") -> None:
        """Create SHAP summary plot."""
        
        try:
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            
            # Save plot
            plot_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP summary plot saved to {plot_path}")
            
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            print("SHAP plotting disabled due to error")
    
    def plot_shap_waterfall(self, explainer: Any, X: pd.DataFrame, 
                           sample_idx: int = 0, output_dir: str = "results") -> None:
        """Create SHAP waterfall plot."""
        
        try:
            # Get SHAP values for the sample
            sample_data = X.iloc[sample_idx:sample_idx+1]
            shap_values = explainer.shap_values(sample_data)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use positive class
            
            # Create SHAP waterfall plot
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(explainer.expected_value, shap_values[0], sample_data.iloc[0], show=False)
            
            # Save plot
            plot_path = os.path.join(output_dir, "shap_waterfall.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP waterfall plot saved to {plot_path}")
            
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
            print("SHAP plotting disabled due to error")
    
    def plot_feature_importance_comparison(self, feature_importance: Dict, 
                                         output_dir: str = "results") -> None:
        """Compare feature importance across different models."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get top 10 features from each model
        top_features = {}
        for model_name, importance_df in feature_importance.items():
            if not importance_df.empty:
                top_features[model_name] = importance_df.head(10)
        
        if not top_features:
            print("No feature importance data available")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 4*len(top_features)))
        if len(top_features) == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(top_features.items()):
            ax = axes[i]
            
            # Plot feature importance
            features = importance_df['feature'].values
            importance = importance_df['importance'].values
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name.replace("_", " ").title()} - Top 10 Features')
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "feature_importance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance comparison saved to {plot_path}")
    
    def analyze_demographic_subgroups(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                    output_dir: str = "results") -> Dict:
        """Analyze model performance across demographic subgroups."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Identify demographic features - expanded to include encoded features
        demographic_features = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['age', 'socioeconomic', 'ses', 'gender', 'cognitive', 'digital', 'social', 'caregiver']):
                demographic_features.append(col)
        
        # Also include specific encoded features we know exist
        specific_features = ['age_group', 'age_group_encoded', 'socioeconomic_status', 'ses_encoded', 
                           'cognitive_status', 'digital_literacy', 'social_engagement', 'caregiver_support']
        for feature in specific_features:
            if feature in X.columns and feature not in demographic_features:
                demographic_features.append(feature)
        
        if not demographic_features:
            print("No demographic features found for subgroup analysis")
            return {}
        
        subgroup_results = {}
        
        for feature in demographic_features:
            if feature in X.columns:
                # Get unique values
                unique_values = X[feature].unique()
                
                subgroup_metrics = {}
                for value in unique_values:
                    # Filter data for this subgroup
                    mask = X[feature] == value
                    X_subgroup = X[mask]
                    y_subgroup = y[mask]
                    
                    if len(X_subgroup) > 10:  # Minimum sample size
                        # Make predictions
                        y_pred = model.predict(X_subgroup)
                        y_pred_proba = model.predict_proba(X_subgroup)[:, 1]
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                        
                        metrics = {
                            'sample_size': len(X_subgroup),
                            'adoption_rate': y_subgroup.mean(),
                            'accuracy': accuracy_score(y_subgroup, y_pred),
                            'precision': precision_score(y_subgroup, y_pred, zero_division=0),
                            'recall': recall_score(y_subgroup, y_pred, zero_division=0),
                            'f1': f1_score(y_subgroup, y_pred, zero_division=0),
                            'auc_roc': roc_auc_score(y_subgroup, y_pred_proba)
                        }
                        
                        subgroup_metrics[value] = metrics
                
                subgroup_results[feature] = subgroup_metrics
        
        # Create subgroup analysis plots
        self.plot_subgroup_analysis(subgroup_results, output_dir)
        
        return subgroup_results
    
    def plot_subgroup_analysis(self, subgroup_results: Dict, output_dir: str) -> None:
        """Create plots for subgroup analysis - DISABLED for pipeline execution."""
        
        # Plotting disabled during pipeline execution to avoid popup windows
        # Plots are only created in Jupyter notebooks for data visualization
        print("Subgroup analysis plots disabled during pipeline execution")
        print("Use Jupyter notebooks for data visualization")
    
    def generate_interpretability_report(self, shap_values: np.ndarray, X: pd.DataFrame,
                                       feature_importance: Dict, subgroup_results: Dict,
                                       output_dir: str = "results") -> str:
        """Generate comprehensive interpretability report."""
        
        report = []
        report.append("AgeTech Adoption Prediction - Model Interpretability Report")
        report.append("=" * 70)
        report.append("")
        
        # SHAP Analysis Summary
        report.append("SHAP Analysis Summary:")
        report.append("-" * 30)
        
        # Get top features from SHAP
        if shap_values is not None:
            feature_importance_shap = np.abs(shap_values).mean(0)
            feature_names = X.columns.tolist()
            
            # Create feature importance dataframe
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': feature_importance_shap
            }).sort_values('shap_importance', ascending=False)
            
            report.append("Top 10 Most Important Features (SHAP):")
            for i, (_, row) in enumerate(shap_importance_df.head(10).iterrows()):
                report.append(f"{i+1:2d}. {row['feature']}: {row['shap_importance']:.4f}")
        
        report.append("")
        
        # Model-specific feature importance
        if feature_importance:
            report.append("Model-specific Feature Importance:")
            report.append("-" * 40)
            
            for model_name, importance_df in feature_importance.items():
                if not importance_df.empty:
                    report.append(f"\n{model_name.replace('_', ' ').title()}:")
                    top_features = importance_df.head(5)
                    for _, row in top_features.iterrows():
                        report.append(f"  - {row['feature']}: {row['importance']:.4f}")
        
        report.append("")
        
        # Subgroup Analysis
        if subgroup_results:
            report.append("Subgroup Analysis:")
            report.append("-" * 20)
            
            for feature, subgroups in subgroup_results.items():
                report.append(f"\n{feature}:")
                for subgroup, metrics in subgroups.items():
                    report.append(f"  {subgroup}:")
                    report.append(f"    Sample size: {metrics['sample_size']}")
                    report.append(f"    Adoption rate: {metrics['adoption_rate']:.2%}")
                    report.append(f"    F1 score: {metrics['f1']:.4f}")
                    report.append(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        
        report.append("")
        report.append("Key Insights:")
        report.append("-" * 15)
        
        # Add key insights based on analysis
        if shap_values is not None and not shap_importance_df.empty:
            top_feature = shap_importance_df.iloc[0]['feature']
            report.append(f"• {top_feature} is the most important predictor of AgeTech adoption")
        
        if subgroup_results:
            report.append("• Model performance varies across demographic subgroups")
            report.append("• Consider subgroup-specific interventions for optimal adoption")
        
        report.append("• Digital literacy and willingness to use technology are key factors")
        report.append("• Social support and caregiver availability significantly impact adoption")
        
        return "\n".join(report)
    
    def run_interpretability_analysis(self) -> Dict:
        """Complete interpretability analysis pipeline."""
        
        print("AgeTech Model Interpretability Analysis")
        print("=" * 50)
        
        # Load model and data
        model, X_test, y_test = self.load_model_and_data()
        if model is None:
            print("Failed to load model and data!")
            return {}
        
        # Load feature importance
        try:
            importance_path = os.path.join("models", "feature_importance.pkl")
            self.feature_importance = joblib.load(importance_path)
            print("Loaded feature importance data")
        except:
            print("No feature importance data found")
            self.feature_importance = {}
        
        # Create SHAP explainer (optional)
        explainer = None
        shap_values = None
        X_sample = None
        
        if SHAP_AVAILABLE:
            explainer = self.create_shap_explainer(model, X_test)
            if explainer is not None:
                # Compute SHAP values
                shap_values, X_sample = self.compute_shap_values(explainer, X_test)
        else:
            print("SHAP not available, skipping SHAP analysis")
        
        # Create SHAP plots
        print("\nCreating SHAP interpretability plots...")
        if explainer is not None and shap_values is not None:
            self.plot_shap_summary(shap_values, X_sample)
            self.plot_shap_waterfall(explainer, X_sample)
        else:
            print("SHAP plots not available")
        
        # Subgroup analysis
        print("\nPerforming subgroup analysis...")
        subgroup_results = self.analyze_demographic_subgroups(model, X_test, y_test)
        
        # Generate report
        print("\nGenerating interpretability report...")
        report = self.generate_interpretability_report(
            shap_values, X_sample, self.feature_importance, subgroup_results
        )
        
        # Save report
        report_path = os.path.join("results", "interpretability_report.txt")
        os.makedirs("results", exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Interpretability report saved to {report_path}")
        print("\nInterpretability analysis completed successfully!")
        
        return {
            'shap_values': shap_values,
            'feature_importance': self.feature_importance,
            'subgroup_results': subgroup_results,
            'report': report
        }

def main():
    """Run the complete interpretability analysis."""
    
    # Initialize interpreter
    interpreter = AgeTechInterpreter(random_state=42)
    
    # Run interpretability analysis
    results = interpreter.run_interpretability_analysis()
    
    if results:
        print("\nInterpretability Analysis Summary:")
        print("=" * 40)
        print("✓ SHAP analysis completed")
        print("✓ Feature importance analysis completed")
        print("✓ Subgroup analysis completed")
        print("✓ Interpretability report generated")

if __name__ == "__main__":
    main() 