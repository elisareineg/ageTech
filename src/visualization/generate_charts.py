"""
AgeTech Visualization Module

This module generates charts and visualizations directly from evaluation 
and interpretability results, saving them as image files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AgeTechVisualizer:
    """
    Generates comprehensive visualizations for AgeTech adoption prediction results.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Use fixed filenames instead of timestamps for easier identification
        self.timestamp = "latest"
        
    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent synthetic data file."""
        try:
            data_files = glob.glob("data/raw/agetch_synthetic_data_*.csv")
            if not data_files:
                print("No synthetic data files found!")
                return pd.DataFrame()
            
            latest_file = max(data_files, key=os.path.getmtime)
            df = pd.read_csv(latest_file)
            print(f"Loaded data from: {latest_file}")
            print(f"Data shape: {df.shape}")
            print(f"Adoption rate: {df['adoption_success'].mean():.2%}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def load_evaluation_results(self) -> Dict:
        """Load evaluation results from saved files."""
        try:
            # Load training results
            if os.path.exists("models/training_results.pkl"):
                import joblib
                training_results = joblib.load("models/training_results.pkl")
                return training_results
            else:
                print("No training results found!")
                return {}
        except Exception as e:
            print(f"Error loading evaluation results: {e}")
            return {}
    
    def load_interpretability_results(self) -> Dict:
        """Load interpretability results from saved files."""
        try:
            # Load feature importance
            if os.path.exists("models/feature_importance.pkl"):
                import joblib
                feature_importance = joblib.load("models/feature_importance.pkl")
                return feature_importance
            else:
                print("No feature importance results found!")
                return {}
        except Exception as e:
            print(f"Error loading interpretability results: {e}")
            return {}
    
    def create_data_overview_charts(self, df: pd.DataFrame):
        """Create data overview and distribution charts."""
        print("Creating data overview charts...")
        
        # 1. Target Variable Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        adoption_counts = df['adoption_success'].value_counts()
        ax1.pie(adoption_counts.values, labels=['Not Adopted', 'Adopted'], 
                autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
        ax1.set_title('AgeTech Adoption Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        adoption_counts.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
        ax2.set_title('AgeTech Adoption Counts', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Adoption Success')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_adoption_distribution_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Demographic Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Age group distribution
        df['age_group'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Age Group Distribution', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Socioeconomic status distribution
        df['socioeconomic_status'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Socioeconomic Status Distribution', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Living situation distribution
        df['living_situation'].value_counts().plot(kind='bar', ax=axes[0,2], color='lightcoral')
        axes[0,2].set_title('Living Situation Distribution', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Adoption by age group
        df.groupby('age_group')['adoption_success'].mean().plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Adoption Rate by Age Group', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylabel('Adoption Rate')
        
        # Adoption by socioeconomic status
        df.groupby('socioeconomic_status')['adoption_success'].mean().plot(kind='bar', ax=axes[1,1], color='lightpink')
        axes[1,1].set_title('Adoption Rate by Socioeconomic Status', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('Adoption Rate')
        
        # Adoption by living situation
        df.groupby('living_situation')['adoption_success'].mean().plot(kind='bar', ax=axes[1,2], color='lightsteelblue')
        axes[1,2].set_title('Adoption Rate by Living Situation', fontweight='bold')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].set_ylabel('Adoption Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_demographic_analysis_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Technology Readiness Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Digital literacy distribution
        df['digital_literacy'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Digital Literacy Distribution', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Internet access distribution
        df['internet_access'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Internet Access Distribution', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Willingness to use new technology
        df['willingness_new_tech'].value_counts().plot(kind='bar', ax=axes[0,2], color='lightcoral')
        axes[0,2].set_title('Willingness to Use New Technology', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Adoption by digital literacy
        df.groupby('digital_literacy')['adoption_success'].mean().plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Adoption Rate by Digital Literacy', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylabel('Adoption Rate')
        
        # Adoption by internet access
        df.groupby('internet_access')['adoption_success'].mean().plot(kind='bar', ax=axes[1,1], color='lightpink')
        axes[1,1].set_title('Adoption Rate by Internet Access', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('Adoption Rate')
        
        # Adoption by willingness
        df.groupby('willingness_new_tech')['adoption_success'].mean().plot(kind='bar', ax=axes[1,2], color='lightsteelblue')
        axes[1,2].set_title('Adoption Rate by Willingness', fontweight='bold')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].set_ylabel('Adoption Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_technology_readiness_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Data overview charts created!")
    
    def create_model_performance_charts(self, training_results: Dict):
        """Create model performance comparison charts."""
        print("Creating model performance charts...")
        
        if not training_results:
            print("No training results available!")
            return
        
        # 1. Model Performance Comparison
        models = []
        f1_scores = []
        auc_scores = []
        accuracy_scores = []
        
        for model_name, results in training_results.items():
            if 'val_metrics' in results:
                models.append(model_name.replace('_', ' ').title())
                f1_scores.append(results['val_metrics'].get('f1', 0))
                auc_scores.append(results['val_metrics'].get('auc_roc', 0))
                accuracy_scores.append(results['val_metrics'].get('accuracy', 0))
        
        if not models:
            print("No valid model results found!")
            return
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # F1 Score comparison
        bars1 = axes[0].bar(models, f1_scores, color='lightblue', alpha=0.8)
        axes[0].set_title('Model F1 Score Comparison', fontweight='bold')
        axes[0].set_ylabel('F1 Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, f1_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AUC Score comparison
        bars2 = axes[1].bar(models, auc_scores, color='lightgreen', alpha=0.8)
        axes[1].set_title('Model AUC-ROC Score Comparison', fontweight='bold')
        axes[1].set_ylabel('AUC-ROC Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
        
        for bar, score in zip(bars2, auc_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy comparison
        bars3 = axes[2].bar(models, accuracy_scores, color='lightcoral', alpha=0.8)
        axes[2].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[2].set_ylabel('Accuracy')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)
        
        for bar, score in zip(bars3, accuracy_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_model_performance_comparison_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Performance Metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a table-like visualization
        metrics_data = []
        for i, model_name in enumerate(models):
            metrics_data.append([
                model_name,
                f"{f1_scores[i]:.3f}",
                f"{auc_scores[i]:.3f}",
                f"{accuracy_scores[i]:.3f}"
            ])
        
        # Create table
        table = ax.table(cellText=metrics_data,
                        colLabels=['Model', 'F1 Score', 'AUC-ROC', 'Accuracy'],
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(metrics_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Detailed Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/05_detailed_performance_metrics_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Model performance charts created!")
    
    def create_feature_importance_charts(self, feature_importance: Dict):
        """Create feature importance visualization charts."""
        print("Creating feature importance charts...")
        
        if not feature_importance:
            print("No feature importance data available!")
            return
        
        # Create feature importance charts for each model
        for model_name, importance_df in feature_importance.items():
            if importance_df.empty:
                continue
            
            # Top 15 features
            top_features = importance_df.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Horizontal bar chart
            y_pos = np.arange(len(top_features))
            bars = ax1.barh(y_pos, top_features['importance'], color='lightblue', alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_features['feature'])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title(f'Top 15 Features - {model_name.replace("_", " ").title()}', 
                         fontweight='bold')
            ax1.invert_yaxis()
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance:.3f}', ha='left', va='center', fontweight='bold')
            
            # Pie chart for top 10 features
            top_10 = top_features.head(10)
            other_importance = top_features.iloc[10:]['importance'].sum()
            
            if other_importance > 0:
                pie_data = list(top_10['importance']) + [other_importance]
                pie_labels = list(top_10['feature']) + ['Others']
            else:
                pie_data = list(top_10['importance'])
                pie_labels = list(top_10['feature'])
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
            wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            ax2.set_title(f'Feature Importance Distribution - {model_name.replace("_", " ").title()}', 
                         fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/06_feature_importance_{model_name}_{self.timestamp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Feature importance charts created!")
    
    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap for numerical features."""
        print("Creating correlation heatmap...")
        
        # Convert categorical to numerical for correlation
        df_numeric = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        # Calculate correlation matrix
        correlation_matrix = df_numeric.corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/07_correlation_heatmap_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top correlations with adoption_success
        adoption_corr = correlation_matrix['adoption_success'].abs().sort_values(ascending=False)
        
        # Create top correlations chart
        plt.figure(figsize=(12, 8))
        top_corr = adoption_corr.head(11)[1:11]  # Exclude adoption_success itself
        
        bars = plt.bar(range(len(top_corr)), top_corr.values, color='lightblue', alpha=0.8)
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation with Adoption Success')
        plt.title('Top 10 Features Correlated with Adoption Success', fontweight='bold')
        plt.xticks(range(len(top_corr)), top_corr.index, rotation=45, ha='right')
        
        # Add value labels
        for bar, corr in zip(bars, top_corr.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/08_top_correlations_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Correlation charts created!")
    
    def create_summary_report(self, df: pd.DataFrame, training_results: Dict, feature_importance: Dict):
        """Create a comprehensive summary report with key insights."""
        print("Creating summary report...")
        
        report = []
        report.append("AgeTech Adoption Prediction - Analysis Summary Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data Overview
        report.append("📊 DATA OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total samples: {len(df)}")
        report.append(f"Adoption rate: {df['adoption_success'].mean():.2%}")
        report.append(f"Features: {len(df.columns) - 1}")  # Exclude target
        report.append("")
        
        # Model Performance Summary
        if training_results:
            report.append("🤖 MODEL PERFORMANCE SUMMARY")
            report.append("-" * 30)
            
            best_model = None
            best_f1 = 0
            
            for model_name, results in training_results.items():
                if 'val_metrics' in results:
                    f1_score = results['val_metrics'].get('f1', 0)
                    auc_score = results['val_metrics'].get('auc_roc', 0)
                    accuracy = results['val_metrics'].get('accuracy', 0)
                    
                    report.append(f"{model_name.replace('_', ' ').title()}:")
                    report.append(f"  F1 Score: {f1_score:.3f}")
                    report.append(f"  AUC-ROC:  {auc_score:.3f}")
                    report.append(f"  Accuracy: {accuracy:.3f}")
                    report.append("")
                    
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = model_name
            
            if best_model:
                report.append(f"🏆 BEST MODEL: {best_model.replace('_', ' ').title()} (F1: {best_f1:.3f})")
                report.append("")
        
        # Key Insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 15)
        
        # Top predictors from correlation
        df_numeric = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        correlation_matrix = df_numeric.corr()
        adoption_corr = correlation_matrix['adoption_success'].abs().sort_values(ascending=False)
        top_predictors = adoption_corr.head(6)[1:6]  # Exclude adoption_success
        
        report.append("Top 5 predictors of AgeTech adoption:")
        for i, (feature, corr) in enumerate(top_predictors.items(), 1):
            report.append(f"  {i}. {feature}: {corr:.3f}")
        report.append("")
        
        # Demographic insights
        report.append("Demographic insights:")
        age_adoption = df.groupby('age_group')['adoption_success'].mean()
        best_age = age_adoption.idxmax()
        worst_age = age_adoption.idxmin()
        report.append(f"  • Highest adoption: {best_age} ({age_adoption[best_age]:.1%})")
        report.append(f"  • Lowest adoption: {worst_age} ({age_adoption[worst_age]:.1%})")
        
        ses_adoption = df.groupby('socioeconomic_status')['adoption_success'].mean()
        best_ses = ses_adoption.idxmax()
        report.append(f"  • Best socioeconomic group: {best_ses} ({ses_adoption[best_ses]:.1%})")
        report.append("")
        
        # Technology insights
        report.append("Technology readiness insights:")
        digital_adoption = df.groupby('digital_literacy')['adoption_success'].mean()
        best_digital = digital_adoption.idxmax()
        report.append(f"  • Digital literacy impact: {best_digital} users ({digital_adoption[best_digital]:.1%} adoption)")
        
        willingness_adoption = df.groupby('willingness_new_tech')['adoption_success'].mean()
        best_willingness = willingness_adoption.idxmax()
        report.append(f"  • Willingness impact: {best_willingness} users ({willingness_adoption[best_willingness]:.1%} adoption)")
        report.append("")
        
        # Save report
        report_path = f"{self.output_dir}/summary_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Summary report saved: {report_path}")
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        print('\n'.join(report))
        
        # Also save with current timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_report_path = f"{self.output_dir}/summary_report_{timestamp}.txt"
        with open(timestamped_report_path, 'w') as f:
            f.write('\n'.join(report))
    
    def cleanup_old_files(self):
        """Clean up old timestamped files, keep only the most recent ones."""
        try:
            # Get all timestamped files (excluding 'latest' files)
            timestamped_files = []
            for file_path in glob.glob(f"{self.output_dir}/*"):
                if "latest" not in file_path and os.path.isfile(file_path):
                    timestamped_files.append(file_path)
            
            if len(timestamped_files) > 11:  # Keep only the most recent set
                # Sort by modification time and keep only the most recent 11 files
                timestamped_files.sort(key=os.path.getmtime, reverse=True)
                files_to_remove = timestamped_files[11:]
                
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                    except:
                        pass
                
                print(f"🧹 Cleaned up {len(files_to_remove)} old timestamped files")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    def cleanup_old_data_files(self):
        """Clean up old data files, keep only the most recent one."""
        try:
            data_dir = "data/raw"
            data_files = glob.glob(f"{data_dir}/agetch_synthetic_data_*.csv")
            
            if len(data_files) <= 1:
                return  # Only one or no files, nothing to clean
            
            # Sort by modification time to find the most recent
            data_files.sort(key=os.path.getmtime, reverse=True)
            most_recent_file = data_files[0]
            files_to_remove = data_files[1:]  # All except the most recent
            
            # Remove old files
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            print(f"🧹 Cleaned up {len(files_to_remove)} old data files")
            print(f"📁 Kept: {os.path.basename(most_recent_file)}")
            
        except Exception as e:
            print(f"⚠️  Data cleanup warning: {e}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations from the latest results."""
        print("🎨 Generating AgeTech Visualizations")
        print("=" * 50)
        
        # Load data and results
        df = self.load_latest_data()
        training_results = self.load_evaluation_results()
        feature_importance = self.load_interpretability_results()
        
        if df.empty:
            print("❌ No data available for visualization!")
            return
        
        # Generate all charts
        self.create_data_overview_charts(df)
        self.create_model_performance_charts(training_results)
        self.create_feature_importance_charts(feature_importance)
        self.create_correlation_heatmap(df)
        self.create_summary_report(df, training_results, feature_importance)
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS COMPLETED!")
        print("="*60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🕒 Current run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # List generated files
        generated_files = glob.glob(f"{self.output_dir}/*{self.timestamp}*")
        print(f"\n📊 Generated {len(generated_files)} visualization files:")
        for file in sorted(generated_files):
            print(f"  • {os.path.basename(file)}")
        
        print(f"\n💡 Tip: Files with 'latest' in the name are always current!")
        print(f"📅 Timestamped versions are also saved for versioning.")
        
        # Clean up old timestamped files (keep only the most recent)
        self.cleanup_old_files()
        
        # Also clean up old data files to keep only the most recent
        self.cleanup_old_data_files()

def main():
    """Run the visualization pipeline."""
    try:
        visualizer = AgeTechVisualizer()
        visualizer.generate_all_visualizations()
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
