"""
Test script for AgeTech data generation functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data.generate_synthetic_data import AgeTechDataGenerator

def test_data_generation():
    """Test the synthetic data generation."""
    print("Testing AgeTech data generation...")
    
    # Initialize generator
    generator = AgeTechDataGenerator(n_samples=100, random_state=42)
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Basic checks
    assert len(df) == 100, f"Expected 100 samples, got {len(df)}"
    assert 'adoption_success' in df.columns, "Missing adoption_success column"
    assert 'participant_id' in df.columns, "Missing participant_id column"
    
    # Check data types
    assert df['adoption_success'].dtype in [np.int64, np.int32], "adoption_success should be integer"
    assert df['participant_id'].dtype in [np.int64, np.int32], "participant_id should be integer"
    
    # Check adoption rate is reasonable
    adoption_rate = df['adoption_success'].mean()
    assert 0.1 <= adoption_rate <= 0.9, f"Adoption rate {adoption_rate:.2%} is outside reasonable range"
    
    # Check for expected columns
    expected_columns = [
        'age_group', 'socioeconomic_status', 'living_situation',
        'cognitive_status', 'physical_mobility', 'hearing_vision_impairment',
        'chronic_conditions', 'medication_effects', 'caregiver_support',
        'social_engagement', 'digital_literacy', 'internet_access',
        'attitude_toward_technology', 'previous_tech_use', 'agetch_experience',
        'tech_assistance_availability', 'willingness_new_tech', 'device_preferences'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"
    
    print("✓ Data generation test passed!")
    print(f"  - Generated {len(df)} samples")
    print(f"  - Adoption rate: {adoption_rate:.2%}")
    print(f"  - Features: {len(df.columns) - 2}")  # Exclude ID and target
    
    return True

def test_feature_engineering():
    """Test the feature engineering components."""
    print("\nTesting feature engineering...")
    
    generator = AgeTechDataGenerator(n_samples=50, random_state=42)
    df = generator.generate_dataset()
    
    # Check composite scores
    assert 'technology_readiness_index' in df.columns, "Missing technology readiness index"
    assert 'health_risk_score' in df.columns, "Missing health risk score"
    
    # Check interaction effects
    assert 'digital_willingness_interaction' in df.columns, "Missing interaction effect"
    
    # Check score ranges
    tech_score = df['technology_readiness_index']
    health_score = df['health_risk_score']
    
    assert tech_score.min() >= 0, "Technology readiness score should be non-negative"
    assert health_score.min() >= 0, "Health risk score should be non-negative"
    
    print("✓ Feature engineering test passed!")
    
    return True

if __name__ == "__main__":
    try:
        test_data_generation()
        test_feature_engineering()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1) 