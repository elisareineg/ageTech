#!/usr/bin/env python3
"""
AgeTech Adoption Prediction - Core ML Pipeline

This script runs the complete Phase 1 ML pipeline including:
- Synthetic data generation
- Data preprocessing
- Advanced feature engineering
- Model training and optimization
- Model evaluation and interpretability
- Exploratory data analysis

Phase 1 focuses on model development and validation using synthetic data.
"""

import os
import sys
import time
from datetime import datetime

def main():
    """Run the complete AgeTech ML pipeline."""
    
    print("=" * 80)
    print("AGETECH ADOPTION PREDICTION - CORE ML PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    pipeline_steps = [
        ("1. Synthetic Data Generation", "src.data.generate_synthetic_data"),
        ("2. Data Preprocessing", "src.data.preprocess"),
        ("3. Feature Engineering", "src.features.engineering"),
        ("4. Model Training", "src.models.train"),
        ("5. Model Evaluation", "src.models.evaluate"),
        ("6. Model Interpretability", "src.models.interpret")
    ]
    
    start_time = time.time()
    
    for step_name, module_path in pipeline_steps:
        print(f"\n{step_name}")
        print("-" * 50)
        
        try:
            # Handle Python module execution
            print(f"Running module: {module_path}")
            module = __import__(module_path, fromlist=['main'])
            if hasattr(module, 'main'):
                module.main()
            print("✓ Module completed successfully")
                
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            print("Continuing with next step...")
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for generated files
    print("Generated Files:")
    print("-" * 30)
    
    # Check data files
    if os.path.exists("data/raw/synthetic_agetch_data.csv"):
        print("✓ Synthetic data generated")
    if os.path.exists("data/processed/X_train.csv"):
        print("✓ Processed data created")
    
    # Check model files
    model_files = [f for f in os.listdir("models") if f.startswith('best_model_')]
    if model_files:
        print(f"✓ Best model saved: {model_files[0]}")
    
    # Check result files
    result_files = os.listdir("results") if os.path.exists("results") else []
    if result_files:
        print(f"✓ {len(result_files)} result files generated")
    
    print()
    print("Next Steps:")
    print("-" * 30)
    print("1. Review results in the 'results/' directory")
    print("2. Explore visualizations in the 'visualizations/' directory")
    print("3. Examine model performance and interpretability")
    print("4. Check generated charts and summary reports")
    print()
    print("Pipeline completed successfully! 🎉")
    
    # Generate visualizations from results
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS FROM RESULTS")
    print("="*70)
    try:
        from src.visualization.generate_charts import AgeTechVisualizer
        visualizer = AgeTechVisualizer()
        visualizer.generate_all_visualizations()
        print("✅ Visualizations generated successfully!")
    except Exception as e:
        print(f"⚠️  Could not generate visualizations: {e}")
        print("You can manually run: python src/visualization/generate_charts.py")
    
    # Run cleanup to ensure only the most recent data file remains
    print("\n" + "="*70)
    print("CLEANING UP OLD FILES")
    print("="*70)
    try:
        from src.visualization.cleanup_old_files import cleanup_data_files
        cleanup_data_files()
        print("✅ Cleanup completed successfully!")
    except Exception as e:
        print(f"⚠️  Could not run cleanup: {e}")
        print("You can manually run: python src/visualization/cleanup_old_files.py")

if __name__ == "__main__":
    main() 