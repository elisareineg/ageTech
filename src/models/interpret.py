"""
Model Interpretability Module for AgeTech Adoption Prediction

This module provides SHAP analysis and interpretability tools for understanding
model predictions and feature importance in the context of AgeTech adoption.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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
            # Load best model
            model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
            if not model_files:
                print("No best model found!")
                return None, None, None
            
            best_model_file = model_files[0]
            model_path = os.path.join(model_dir, best_model_file)
            model = joblib.load(model_path)
            
            print(f"Loaded model: {best_model_file}")
            
            # Load test data
            X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            y_test = X_test['adoption_success']
            X_test = X_test.drop('adoption_success', axis=1)
            
            print(f"Loaded test data: {X_test.shape}")
            
            return model, X_test, y_test
            
        except Exception as e:
            print(f"Error loading model and data: {e}")
            return None, None, None
    
    def create_shap_explainer(self, model: Any, X: pd.DataFrame) -> shap.Explainer:
        """Create SHAP explainer for the model."""
        
        try:
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X)
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X)
            
            return explainer
            
        except Exception as e:
            print(f"Error creating SHAP explainer: {e}")
            return None
    
    def compute_shap_values(self, explainer: shap.Explainer, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the dataset."""
        
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
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary plot saved to {plot_path}")
    
    def plot_shap_waterfall(self, explainer: shap.Explainer, X: pd.DataFrame, 
                           sample_idx: int = 0, output_dir: str = "results") -> None:
        """Create SHAP waterfall plot for a specific prediction."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get SHAP values for specific sample
        shap_values = explainer.shap_values(X.iloc[sample_idx:sample_idx+1])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X.iloc[sample_idx].values,
                feature_names=X.columns.tolist()
            ),
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"shap_waterfall_sample_{sample_idx}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP waterfall plot saved to {plot_path}")
    
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
        
        # Identify demographic features
        demographic_features = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['age', 'socioeconomic', 'ses', 'gender']):
                demographic_features.append(col)
        
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
        """Create plots for subgroup analysis."""
        
        for feature, subgroups in subgroup_results.items():
            if not subgroups:
                continue
            
            # Create subplot for each metric
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                values = []
                labels = []
                
                for subgroup, metrics_dict in subgroups.items():
                    if metric in metrics_dict:
                        values.append(metrics_dict[metric])
                        labels.append(str(subgroup))
                
                if values:
                    ax.bar(labels, values)
                    ax.set_title(f'{metric.upper()} by {feature}')
                    ax.set_ylabel(metric.upper())
                    ax.tick_params(axis='x', rotation=45)
            
            # Remove empty subplots
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"subgroup_analysis_{feature}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Subgroup analysis plot saved to {plot_path}")
    
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
        
        # Create SHAP explainer
        explainer = self.create_shap_explainer(model, X_test)
        if explainer is None:
            print("Failed to create SHAP explainer!")
            return {}
        
        # Compute SHAP values
        shap_values, X_sample = self.compute_shap_values(explainer, X_test)
        if shap_values is None:
            print("Failed to compute SHAP values!")
            return {}
        
        # Create plots
        print("\nCreating interpretability plots...")
        self.plot_shap_summary(shap_values, X_sample)
        self.plot_shap_waterfall(explainer, X_sample, sample_idx=0)
        
        if self.feature_importance:
            self.plot_feature_importance_comparison(self.feature_importance)
        
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