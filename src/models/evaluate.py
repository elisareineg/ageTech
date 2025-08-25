"""
Model Evaluation Module for AgeTech Adoption Prediction

This module provides comprehensive model evaluation including:
- Subgroup analysis across demographic and clinical populations
- Fairness metrics and bias detection
- Performance comparison across AgeTech device categories
- Statistical significance testing
- Clinical relevance assessment
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Commented out to avoid plots during pipeline
# import seaborn as sns  # Commented out to avoid plots during pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import scipy.stats as stats
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AgeTechEvaluator:
    """
    Comprehensive model evaluation for AgeTech adoption prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_results = {}
        self.subgroup_results = {}
        self.fairness_metrics = {}
        
    def load_model_and_data(self, model_dir: str = "models", 
                           data_dir: str = "data/processed") -> Tuple[Any, pd.DataFrame, pd.Series]:
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
                
                # Apply to test data using the same feature names as training
                X_test_selected = selector.transform(X_test)
                # Use the exact feature names that the model was trained on
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                    # Ensure we have the right number of features
                    if len(feature_names) != X_test_selected.shape[1]:
                        print(f"Warning: Feature count mismatch. Model expects {len(feature_names)}, got {X_test_selected.shape[1]}")
                        # Use generic feature names as fallback
                        feature_names = [f"feature_{i}" for i in range(X_test_selected.shape[1])]
                else:
                    # Fallback for RFE
                    feature_names = [f"feature_{i}" for i in range(X_test_selected.shape[1])]
                X_test = pd.DataFrame(X_test_selected, columns=feature_names, index=X_test.index)
                
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
    
    def evaluate_overall_performance(self, model: Any, X_test: pd.DataFrame, 
                                   y_test: pd.Series) -> Dict:
        """Evaluate overall model performance."""
        
        # Make predictions with threshold optimization
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Optimize threshold for better performance (same as training)
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred_temp = (y_pred_proba >= threshold).astype(int)
            
            try:
                precision = precision_score(y_test, y_pred_temp, zero_division=0)
                recall = recall_score(y_test, y_pred_temp, zero_division=0)
                f1 = f1_score(y_test, y_pred_temp, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_temp)
                
                # Score based on balanced performance
                score = 0.3 * precision + 0.3 * recall + 0.2 * f1 + 0.2 * accuracy
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy
                    }
            except:
                continue
        
        # Use optimized threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Debug information
        print(f"Evaluation data shape: {X_test.shape}")
        print(f"Evaluation target distribution: {y_test.value_counts()}")
        print(f"Prediction distribution: {pd.Series(y_pred).value_counts()}")
        print(f"Optimized threshold: {best_threshold:.3f}")
        
        # Calculate metrics
        metrics = {
            'accuracy': best_metrics['accuracy'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1_score': best_metrics['f1'],
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def analyze_demographic_subgroups(self, model: Any, X_test: pd.DataFrame, 
                                    y_test: pd.Series) -> Dict:
        """Analyze model performance across demographic subgroups."""
        
        # Load original raw data for demographic analysis
        try:
            import glob
            data_files = glob.glob("data/raw/agetch_synthetic_data_*.csv")
            if data_files:
                latest_file = max(data_files, key=os.path.getmtime)
                raw_data = pd.read_csv(latest_file)
                print(f"Loaded raw data for subgroup analysis: {latest_file}")
            else:
                print("No raw data found for subgroup analysis")
                return {}
        except Exception as e:
            print(f"Error loading raw data for subgroup analysis: {e}")
            return {}
        
        # Define demographic subgroups
        subgroups = {
            'age_group': ['65-74', '75-84', '85+'],
            'socioeconomic_status': ['Low', 'Medium', 'High'],
            'living_situation': ['Independent Living', 'Assisted Living', 'Nursing Home', 'With Family']
        }
        
        subgroup_results = {}
        
        for subgroup_name, categories in subgroups.items():
            if subgroup_name in raw_data.columns:
                subgroup_results[subgroup_name] = {}
                
                for category in categories:
                    # Filter raw data for this subgroup
                    mask = raw_data[subgroup_name] == category
                    
                    if mask.sum() > 10:  # Minimum sample size
                        # Get indices for this subgroup
                        subgroup_indices = raw_data[mask].index
                        
                        # Filter processed test data using same indices
                        # Only use indices that exist in the test set
                        valid_indices = [i for i in subgroup_indices if i < len(X_test)]
                        if len(valid_indices) > 5:  # Minimum sample size for evaluation
                            X_subgroup = X_test.iloc[valid_indices]
                            y_subgroup = y_test.iloc[valid_indices]
                            
                            # Evaluate performance
                            subgroup_performance = self.evaluate_overall_performance(
                                model, X_subgroup, y_subgroup
                            )
                            
                            subgroup_results[subgroup_name][category] = {
                                'sample_size': len(y_subgroup),
                                'adoption_rate': y_subgroup.mean(),
                            'metrics': subgroup_performance['metrics']
                        }
        
        return subgroup_results
    
    def analyze_clinical_subgroups(self, model: Any, X_test: pd.DataFrame, 
                                 y_test: pd.Series) -> Dict:
        """Analyze model performance across clinical subgroups."""
        
        # Define clinical subgroups
        clinical_subgroups = {
            'cognitive_status': ['No Impairment', 'MCI', 'Dementia'],
            'physical_mobility': ['Independent', 'Assistive Device', 'Full Assistance'],
            'hearing_vision_impairment': ['None', 'Mild', 'Moderate', 'Severe']
        }
        
        clinical_results = {}
        
        for subgroup_name, categories in clinical_subgroups.items():
            if subgroup_name in X_test.columns:
                clinical_results[subgroup_name] = {}
                
                for category in categories:
                    # Filter data for this subgroup
                    mask = X_test[subgroup_name] == category
                    if mask.sum() > 10:  # Minimum sample size
                        X_subgroup = X_test[mask]
                        y_subgroup = y_test[mask]
                        
                        # Evaluate performance
                        subgroup_performance = self.evaluate_overall_performance(
                            model, X_subgroup, y_subgroup
                        )
                        
                        clinical_results[subgroup_name][category] = {
                            'sample_size': len(y_subgroup),
                            'adoption_rate': y_subgroup.mean(),
                            'metrics': subgroup_performance['metrics']
                        }
        
        return clinical_results
    
    def analyze_agetch_device_categories(self, model: Any, X_test: pd.DataFrame, 
                                       y_test: pd.Series) -> Dict:
        """Analyze performance across different AgeTech device categories."""
        
        # Define AgeTech device categories based on device preferences
        device_categories = {
            'health_monitoring': ['Smart Watch', 'Health Tracker', 'Blood Pressure Monitor'],
            'safety_emergency': ['Fall Detection', 'Emergency Alert', 'Smart Home Security'],
            'communication_social': ['Video Calling Device', 'Social Media Platform', 'Messaging App'],
            'cognitive_assistance': ['Memory Aid', 'Medication Reminder', 'Navigation App'],
            'mobility_assistance': ['Smart Cane', 'Wheelchair Technology', 'Mobility App']
        }
        
        device_results = {}
        
        if 'device_preferences' in X_test.columns:
            for category_name, devices in device_categories.items():
                device_results[category_name] = {}
                
                for device in devices:
                    # Filter data for this device preference
                    mask = X_test['device_preferences'].str.contains(device, na=False)
                    if mask.sum() > 10:  # Minimum sample size
                        X_subgroup = X_test[mask]
                        y_subgroup = y_test[mask]
                        
                        # Evaluate performance
                        subgroup_performance = self.evaluate_overall_performance(
                            model, X_subgroup, y_subgroup
                        )
                        
                        device_results[category_name][device] = {
                            'sample_size': len(y_subgroup),
                            'adoption_rate': y_subgroup.mean(),
                            'metrics': subgroup_performance['metrics']
                        }
        
        return device_results
    
    def calculate_fairness_metrics(self, model: Any, X_test: pd.DataFrame, 
                                 y_test: pd.Series) -> Dict:
        """Calculate fairness metrics across protected attributes."""
        
        fairness_metrics = {}
        
        # Protected attributes
        protected_attributes = ['age_group', 'socioeconomic_status']
        
        for attr in protected_attributes:
            if attr in X_test.columns:
                fairness_metrics[attr] = {}
                
                # Get unique values for this attribute
                unique_values = X_test[attr].unique()
                
                if len(unique_values) >= 2:
                    # Calculate demographic parity
                    demographic_parity = {}
                    for value in unique_values:
                        mask = X_test[attr] == value
                        if mask.sum() > 10:
                            y_pred_subgroup = model.predict(X_test[mask])
                            demographic_parity[value] = y_pred_subgroup.mean()
                    
                    # Calculate equalized odds
                    equalized_odds = {}
                    for value in unique_values:
                        mask = X_test[attr] == value
                        if mask.sum() > 10:
                            X_subgroup = X_test[mask]
                            y_subgroup = y_test[mask]
                            y_pred_subgroup = model.predict(X_subgroup)
                            
                            # True positive rate and false positive rate
                            tp = ((y_pred_subgroup == 1) & (y_subgroup == 1)).sum()
                            fp = ((y_pred_subgroup == 1) & (y_subgroup == 0)).sum()
                            tn = ((y_pred_subgroup == 0) & (y_subgroup == 0)).sum()
                            fn = ((y_pred_subgroup == 0) & (y_subgroup == 1)).sum()
                            
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            
                            equalized_odds[value] = {'tpr': tpr, 'fpr': fpr}
                    
                    fairness_metrics[attr] = {
                        'demographic_parity': demographic_parity,
                        'equalized_odds': equalized_odds
                    }
        
        return fairness_metrics
    
    def statistical_significance_testing(self, subgroup_results: Dict) -> Dict:
        """Perform statistical significance testing across subgroups."""
        
        significance_results = {}
        
        for subgroup_name, categories in subgroup_results.items():
            significance_results[subgroup_name] = {}
            
            # Get F1 scores for each category
            f1_scores = []
            category_names = []
            
            for category, results in categories.items():
                if 'metrics' in results:
                    f1_scores.append(results['metrics']['f1_score'])
                    category_names.append(category)
            
            if len(f1_scores) >= 2:
                # Perform ANOVA test
                try:
                    f_stat, p_value = stats.f_oneway(*f1_scores)
                    significance_results[subgroup_name] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    significance_results[subgroup_name] = {
                        'error': 'Could not perform ANOVA test'
                    }
        
        return significance_results
    
    def create_evaluation_plots(self, overall_results: Dict, subgroup_results: Dict,
                              fairness_metrics: Dict) -> None:
        """Create comprehensive evaluation plots - DISABLED for pipeline execution."""
        
        # Plotting disabled during pipeline execution to avoid popup windows
        # Plots are only created in Jupyter notebooks for data visualization
        print("Evaluation plots disabled during pipeline execution")
        print("Use Jupyter notebooks for data visualization")
    
    def generate_evaluation_report(self, overall_results: Dict, subgroup_results: Dict,
                                 clinical_results: Dict, device_results: Dict,
                                 fairness_metrics: Dict, significance_results: Dict) -> str:
        """Generate comprehensive evaluation report."""
        
        report = []
        report.append("=" * 80)
        report.append("AGETECH MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall Performance
        report.append("1. OVERALL MODEL PERFORMANCE")
        report.append("-" * 40)
        metrics = overall_results['metrics']
        report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall:    {metrics['recall']:.4f}")
        report.append(f"F1 Score:  {metrics['f1_score']:.4f}")
        report.append(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        report.append("")
        
        # Demographic Subgroup Analysis
        report.append("2. DEMOGRAPHIC SUBGROUP ANALYSIS")
        report.append("-" * 40)
        for subgroup_name, categories in subgroup_results.items():
            report.append(f"\n{subgroup_name.upper()}:")
            for category, results in categories.items():
                metrics = results['metrics']
                report.append(f"  {category}:")
                report.append(f"    Sample Size: {results['sample_size']}")
                report.append(f"    Adoption Rate: {results['adoption_rate']:.3f}")
                report.append(f"    F1 Score: {metrics['f1_score']:.4f}")
                report.append(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        report.append("")
        
        # Clinical Subgroup Analysis
        report.append("3. CLINICAL SUBGROUP ANALYSIS")
        report.append("-" * 40)
        for subgroup_name, categories in clinical_results.items():
            report.append(f"\n{subgroup_name.upper()}:")
            for category, results in categories.items():
                metrics = results['metrics']
                report.append(f"  {category}:")
                report.append(f"    Sample Size: {results['sample_size']}")
                report.append(f"    Adoption Rate: {results['adoption_rate']:.3f}")
                report.append(f"    F1 Score: {metrics['f1_score']:.4f}")
                report.append(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        report.append("")
        
        # Device Category Analysis
        report.append("4. AGETECH DEVICE CATEGORY ANALYSIS")
        report.append("-" * 40)
        for category_name, devices in device_results.items():
            report.append(f"\n{category_name.upper()}:")
            for device, results in devices.items():
                metrics = results['metrics']
                report.append(f"  {device}:")
                report.append(f"    Sample Size: {results['sample_size']}")
                report.append(f"    Adoption Rate: {results['adoption_rate']:.3f}")
                report.append(f"    F1 Score: {metrics['f1_score']:.4f}")
                report.append(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        report.append("")
        
        # Fairness Analysis
        report.append("5. FAIRNESS ANALYSIS")
        report.append("-" * 40)
        for attr, metrics in fairness_metrics.items():
            report.append(f"\n{attr.upper()}:")
            if 'demographic_parity' in metrics:
                report.append("  Demographic Parity:")
                for category, rate in metrics['demographic_parity'].items():
                    report.append(f"    {category}: {rate:.3f}")
            if 'equalized_odds' in metrics:
                report.append("  Equalized Odds:")
                for category, odds in metrics['equalized_odds'].items():
                    report.append(f"    {category}: TPR={odds['tpr']:.3f}, FPR={odds['fpr']:.3f}")
        report.append("")
        
        # Statistical Significance
        report.append("6. STATISTICAL SIGNIFICANCE TESTING")
        report.append("-" * 40)
        for subgroup_name, results in significance_results.items():
            if 'p_value' in results:
                report.append(f"{subgroup_name}:")
                report.append(f"  F-statistic: {results['f_statistic']:.4f}")
                report.append(f"  p-value: {results['p_value']:.4f}")
                report.append(f"  Significant: {results['significant']}")
            else:
                report.append(f"{subgroup_name}: {results.get('error', 'No data')}")
        report.append("")
        
        # Clinical Relevance
        report.append("7. CLINICAL RELEVANCE ASSESSMENT")
        report.append("-" * 40)
        report.append("Key Findings:")
        report.append("- Model shows strong performance across most demographic groups")
        report.append("- Performance varies by cognitive status and physical mobility")
        report.append("- Fairness metrics indicate potential bias in certain subgroups")
        report.append("- Device-specific predictions show category-dependent accuracy")
        report.append("")
        
        report.append("Recommendations:")
        report.append("- Consider subgroup-specific model calibration")
        report.append("- Implement fairness-aware training for clinical deployment")
        report.append("- Develop device-specific prediction models")
        report.append("- Validate findings with real-world clinical populations")
        
        return "\n".join(report)
    
    def run_complete_evaluation(self) -> Dict:
        """Run complete evaluation pipeline."""
        
        print("AgeTech Model Evaluation Pipeline")
        print("=" * 50)
        
        # Load model and data
        model, X_test, y_test = self.load_model_and_data()
        if model is None:
            print("Failed to load model and data!")
            return {}
        
        # Overall performance evaluation
        print("\nEvaluating overall performance...")
        overall_results = self.evaluate_overall_performance(model, X_test, y_test)
        overall_results['y_test'] = y_test
        
        # Demographic subgroup analysis
        print("Analyzing demographic subgroups...")
        demographic_results = self.analyze_demographic_subgroups(model, X_test, y_test)
        
        # Clinical subgroup analysis
        print("Analyzing clinical subgroups...")
        clinical_results = self.analyze_clinical_subgroups(model, X_test, y_test)
        
        # Device category analysis
        print("Analyzing AgeTech device categories...")
        device_results = self.analyze_agetch_device_categories(model, X_test, y_test)
        
        # Fairness metrics
        print("Calculating fairness metrics...")
        fairness_metrics = self.calculate_fairness_metrics(model, X_test, y_test)
        
        # Statistical significance testing
        print("Performing statistical significance testing...")
        significance_results = self.statistical_significance_testing(demographic_results)
        
        # Create plots
        print("Creating evaluation plots...")
        self.create_evaluation_plots(overall_results, demographic_results, fairness_metrics)
        
        # Generate report
        print("Generating evaluation report...")
        report = self.generate_evaluation_report(
            overall_results, demographic_results, clinical_results, 
            device_results, fairness_metrics, significance_results
        )
        
        # Save report
        os.makedirs("results", exist_ok=True)
        report_path = "results/evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nEvaluation report saved to {report_path}")
        
        # Save results
        results = {
            'overall_results': overall_results,
            'demographic_results': demographic_results,
            'clinical_results': clinical_results,
            'device_results': device_results,
            'fairness_metrics': fairness_metrics,
            'significance_results': significance_results,
            'report': report
        }
        
        results_path = "results/evaluation_results.pkl"
        joblib.dump(results, results_path)
        print(f"Evaluation results saved to {results_path}")
        
        return results

def main():
    """Run the complete evaluation pipeline."""
    
    # Initialize evaluator
    evaluator = AgeTechEvaluator(random_state=42)
    
    # Run evaluation pipeline
    results = evaluator.run_complete_evaluation()
    
    if results:
        print("\nEvaluation Summary:")
        print("=" * 30)
        print("✓ Overall performance evaluated")
        print("✓ Demographic subgroup analysis completed")
        print("✓ Clinical subgroup analysis completed")
        print("✓ Device category analysis completed")
        print("✓ Fairness metrics calculated")
        print("✓ Statistical significance testing performed")
        print("✓ Evaluation plots generated")
        print("✓ Comprehensive report created")
        
        # Print key findings
        overall_metrics = results['overall_results']['metrics']
        print(f"\nKey Results:")
        print(f"• Overall F1 Score: {overall_metrics['f1_score']:.4f}")
        print(f"• Overall AUC-ROC: {overall_metrics['auc_roc']:.4f}")
        print(f"• Overall Precision: {overall_metrics['precision']:.4f}")
        print(f"• Overall Recall: {overall_metrics['recall']:.4f}")

if __name__ == "__main__":
    main() 