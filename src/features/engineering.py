"""
Feature Engineering for AgeTech Adoption Prediction

This module implements feature engineering techniques including:
- Composite scoring for digital literacy and technology readiness
- Interaction terms between top predictors
- Technology readiness indices
- Health risk scoring
- Social support scoring
- Advanced feature transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class AgeTechFeatureEngineer:
    """
    Advanced feature engineering for AgeTech adoption prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.scalers = {}
        self.feature_importance = {}
        self.composite_scores = {}
        
    def create_digital_literacy_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Create composite digital literacy score based on CPQ framework.
        
        Components:
        - Digital literacy level
        - Internet access
        - Previous tech use
        - AgeTech experience
        - Tech assistance availability
        """
        
        # Define weights for digital literacy components
        weights = {
            'digital_literacy': 0.25,
            'internet_access': 0.20,
            'previous_tech_use': 0.20,
            'agetch_experience': 0.20,
            'tech_assistance_availability': 0.15
        }
        
        # Create numeric mappings
        literacy_mapping = {
            'Basic': 1, 'Intermediate': 2, 'Advanced': 3
        }
        
        internet_mapping = {
            'No Access': 0, 'Limited': 1, 'Reliable': 2, 'High Speed': 3
        }
        
        tech_use_mapping = {
            'None': 0, 'Basic': 1, 'Moderate': 2, 'Extensive': 3
        }
        
        experience_mapping = {
            'None': 0, 'Some': 1, 'Moderate': 2, 'Extensive': 3
        }
        
        assistance_mapping = {
            'None': 0, 'Limited': 1, 'Available': 2, 'Excellent': 3
        }
        
        # Calculate weighted score
        score = (
            df['digital_literacy'].map(literacy_mapping) * weights['digital_literacy'] +
            df['internet_access'].map(internet_mapping) * weights['internet_access'] +
            df['previous_tech_use'].map(tech_use_mapping) * weights['previous_tech_use'] +
            df['agetch_experience'].map(experience_mapping) * weights['agetch_experience'] +
            df['tech_assistance_availability'].map(assistance_mapping) * weights['tech_assistance_availability']
        )
        
        # Normalize to 0-1 scale
        score = (score - score.min()) / (score.max() - score.min())
        
        # Add interaction terms for better predictive power
        df['digital_literacy_score'] = score
        
        # Create attitude mapping for interactions
        attitude_mapping = {
            'Negative': 0, 'Neutral': 0.5, 'Positive': 1
        }
        
        df['digital_attitude_interaction'] = (
            score * df['attitude_toward_technology'].map(attitude_mapping)
        )
        df['tech_access_interaction'] = (
            df['internet_access'].map(internet_mapping) * 
            df['previous_tech_use'].map(tech_use_mapping)
        )
        
        return score
    
    def create_technology_readiness_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Create technology readiness index based on TAQ-12 framework.
        
        Components:
        - Digital literacy score
        - Attitude toward technology
        - Willingness to use new technology
        - Previous technology experience
        """
        
        # Get digital literacy score
        digital_literacy_score = self.create_digital_literacy_score(df)
        
        # Create attitude mapping
        attitude_mapping = {
            'Negative': 0, 'Neutral': 0.5, 'Positive': 1
        }
        
        willingness_mapping = {
            'Unwilling': 0, 'Neutral': 0.5, 'Willing': 1
        }
        
        # Calculate technology readiness index
        tri_score = (
            digital_literacy_score * 0.4 +
            df['attitude_toward_technology'].map(attitude_mapping) * 0.3 +
            df['willingness_new_tech'].map(willingness_mapping) * 0.3
        )
        
        # Add enhanced interaction terms
        df['technology_readiness_index'] = tri_score
        df['tech_willingness_interaction'] = (
            tri_score * df['willingness_new_tech'].map(willingness_mapping)
        )
        df['literacy_attitude_interaction'] = (
            digital_literacy_score * df['attitude_toward_technology'].map(attitude_mapping)
        )
        
        # Add polynomial features for better non-linear relationships
        df['tech_readiness_squared'] = tri_score ** 2
        df['digital_literacy_squared'] = digital_literacy_score ** 2
        
        # Add ratio features
        df['tech_to_health_ratio'] = tri_score / (df['health_risk_score'] + 1e-6)
        df['literacy_to_social_ratio'] = digital_literacy_score / (df['social_support_score'] + 1e-6)
        
        return tri_score
    
    def create_health_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Create composite health risk score.
        
        Components:
        - Cognitive status
        - Physical mobility
        - Hearing/vision impairment
        - Chronic conditions
        - Medication effects
        """
        
        # Define weights for health components
        weights = {
            'cognitive_status': 0.30,
            'physical_mobility': 0.25,
            'hearing_vision_impairment': 0.20,
            'chronic_conditions': 0.15,
            'medication_effects': 0.10
        }
        
        # Create numeric mappings
        cognitive_mapping = {
            'No Impairment': 0, 'MCI': 1, 'Dementia': 2
        }
        
        mobility_mapping = {
            'Independent': 0, 'Assistive Device': 1, 'Full Assistance': 2
        }
        
        impairment_mapping = {
            'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3
        }
        
        conditions_mapping = {
            'None': 0, '1-2': 1, '3-4': 2, '5+': 3
        }
        
        medication_mapping = {
            'None': 0, '1-2': 1, '3-4': 2, '5+': 3
        }
        
        # Calculate weighted health risk score
        health_risk = (
            df['cognitive_status'].map(cognitive_mapping) * weights['cognitive_status'] +
            df['physical_mobility'].map(mobility_mapping) * weights['physical_mobility'] +
            df['hearing_vision_impairment'].map(impairment_mapping) * weights['hearing_vision_impairment'] +
            df['chronic_conditions'].map(conditions_mapping) * weights['chronic_conditions'] +
            df['medication_effects'].map(medication_mapping) * weights['medication_effects']
        )
        
        # Normalize to 0-1 scale (higher = higher risk)
        health_risk = (health_risk - health_risk.min()) / (health_risk.max() - health_risk.min())
        
        return health_risk
    
    def create_social_support_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Create composite social support score.
        
        Components:
        - Caregiver support availability
        - Social engagement level
        - Tech assistance availability
        """
        
        # Define weights
        weights = {
            'caregiver_support': 0.40,
            'social_engagement': 0.35,
            'tech_assistance': 0.25
        }
        
        # Create mappings
        caregiver_mapping = {
            'None': 0, 'Limited': 1, 'Available': 2, 'Excellent': 3
        }
        
        engagement_mapping = {
            'Isolated': 0, 'Limited': 1, 'Moderate': 2, 'Active': 3
        }
        
        assistance_mapping = {
            'None': 0, 'Limited': 1, 'Available': 2, 'Excellent': 3
        }
        
        # Calculate social support score
        social_support = (
            df['caregiver_support'].map(caregiver_mapping) * weights['caregiver_support'] +
            df['social_engagement'].map(engagement_mapping) * weights['social_engagement'] +
            df['tech_assistance_availability'].map(assistance_mapping) * weights['tech_assistance']
        )
        
        # Normalize to 0-1 scale
        social_support = (social_support - social_support.min()) / (social_support.max() - social_support.min())
        
        return social_support
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between top predictors.
        
        Key interactions:
        - Digital Literacy × Willingness to Use Technology
        - Technology Readiness × Social Support
        - Health Risk × Technology Readiness
        - Age Group × Digital Literacy
        """
        
        # Create composite scores first
        df['digital_literacy_score'] = self.create_digital_literacy_score(df)
        df['technology_readiness_index'] = self.create_technology_readiness_index(df)
        df['health_risk_score'] = self.create_health_risk_score(df)
        df['social_support_score'] = self.create_social_support_score(df)
        
        # Create age group mapping
        age_mapping = {
            '65-74': 0, '75-84': 1, '85+': 2
        }
        df['age_group_numeric'] = df['age_group'].map(age_mapping)
        
        # Create interaction features
        df['digital_literacy_willingness_interaction'] = (
            df['digital_literacy_score'] * 
            df['willingness_new_tech'].map({'Unwilling': 0, 'Neutral': 0.5, 'Willing': 1})
        )
        
        df['tech_readiness_social_support_interaction'] = (
            df['technology_readiness_index'] * df['social_support_score']
        )
        
        df['health_risk_tech_readiness_interaction'] = (
            df['health_risk_score'] * df['technology_readiness_index']
        )
        
        df['age_digital_literacy_interaction'] = (
            df['age_group_numeric'] * df['digital_literacy_score']
        )
        
        # Create polynomial features for key variables
        df['digital_literacy_squared'] = df['digital_literacy_score'] ** 2
        df['tech_readiness_squared'] = df['technology_readiness_index'] ** 2
        
        return df
    
    def create_device_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on AgeTech device categories.
        """
        
        # Define device categories
        device_categories = {
            'health_monitoring': ['Smart Watch', 'Health Tracker', 'Blood Pressure Monitor'],
            'safety_emergency': ['Fall Detection', 'Emergency Alert', 'Smart Home Security'],
            'communication_social': ['Video Calling Device', 'Social Media Platform', 'Messaging App'],
            'cognitive_assistance': ['Memory Aid', 'Medication Reminder', 'Navigation App'],
            'mobility_assistance': ['Smart Cane', 'Wheelchair Technology', 'Mobility App']
        }
        
        # Create binary features for each category
        for category_name, devices in device_categories.items():
            df[f'prefers_{category_name}'] = df['device_preferences'].apply(
                lambda x: any(device in str(x) for device in devices)
            ).astype(int)
        
        # Create device complexity score
        complexity_mapping = {
            'Smart Watch': 3, 'Health Tracker': 2, 'Blood Pressure Monitor': 1,
            'Fall Detection': 2, 'Emergency Alert': 1, 'Smart Home Security': 3,
            'Video Calling Device': 2, 'Social Media Platform': 3, 'Messaging App': 1,
            'Memory Aid': 1, 'Medication Reminder': 1, 'Navigation App': 2,
            'Smart Cane': 1, 'Wheelchair Technology': 2, 'Mobility App': 2
        }
        
        # Calculate average device complexity preference
        def calculate_complexity_preference(preferences):
            if pd.isna(preferences):
                return 0
            devices = str(preferences).split(',')
            complexities = [complexity_mapping.get(device.strip(), 0) for device in devices]
            return np.mean(complexities) if complexities else 0
        
        df['device_complexity_preference'] = df['device_preferences'].apply(calculate_complexity_preference)
        
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced demographic features.
        """
        
        # Create socioeconomic status mapping
        ses_mapping = {
            'Low': 0, 'Medium': 1, 'High': 2
        }
        df['socioeconomic_status_numeric'] = df['socioeconomic_status'].map(ses_mapping)
        
        # Create living situation mapping
        living_mapping = {
            'Independent Living': 0, 'Assisted Living': 1, 'Nursing Home': 2, 'With Family': 1
        }
        df['living_situation_numeric'] = df['living_situation'].map(living_mapping)
        
        # Create age-related features
        df['is_young_old'] = (df['age_group'] == '65-74').astype(int)
        df['is_old_old'] = (df['age_group'] == '85+').astype(int)
        
        # Create socioeconomic interaction features
        df['age_ses_interaction'] = df['age_group_numeric'] * df['socioeconomic_status_numeric']
        df['living_ses_interaction'] = df['living_situation_numeric'] * df['socioeconomic_status_numeric']
        
        return df
    
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced clinical features.
        """
        
        # Create cognitive status mapping
        cognitive_mapping = {
            'No Impairment': 0, 'MCI': 1, 'Dementia': 2
        }
        df['cognitive_status_numeric'] = df['cognitive_status'].map(cognitive_mapping)
        
        # Create physical mobility mapping
        mobility_mapping = {
            'Independent': 0, 'Assistive Device': 1, 'Full Assistance': 2
        }
        df['physical_mobility_numeric'] = df['physical_mobility'].map(mobility_mapping)
        
        # Create impairment severity score
        impairment_mapping = {
            'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3
        }
        df['hearing_vision_impairment_numeric'] = df['hearing_vision_impairment'].map(impairment_mapping)
        
        # Create clinical risk score
        df['clinical_risk_score'] = (
            df['cognitive_status_numeric'] * 0.4 +
            df['physical_mobility_numeric'] * 0.3 +
            df['hearing_vision_impairment_numeric'] * 0.3
        )
        
        # Create clinical interaction features
        df['cognitive_mobility_interaction'] = (
            df['cognitive_status_numeric'] * df['physical_mobility_numeric']
        )
        
        df['cognitive_impairment_interaction'] = (
            df['cognitive_status_numeric'] * df['hearing_vision_impairment_numeric']
        )
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target: pd.Series, 
                           method: str = 'mutual_info', k: int = 20) -> pd.DataFrame:
        """
        Select the best features using statistical methods.
        """
        
        # Separate numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = df[numeric_features]
        
        # Remove target variable if present
        if 'adoption_success' in X_numeric.columns:
            X_numeric = X_numeric.drop('adoption_success', axis=1)
        
        # Select best features
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_numeric.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_numeric.shape[1]))
        
        X_selected = selector.fit_transform(X_numeric, target)
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        self.feature_importance = dict(zip(X_numeric.columns, selector.scores_))
        
        # Return DataFrame with selected features
        return df[selected_features + ['adoption_success'] if 'adoption_success' in df.columns else selected_features]
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features for AgeTech adoption prediction.
        """
        
        print("Creating advanced features...")
        
        # Create composite scores
        df['digital_literacy_score'] = self.create_digital_literacy_score(df)
        df['technology_readiness_index'] = self.create_technology_readiness_index(df)
        df['health_risk_score'] = self.create_health_risk_score(df)
        df['social_support_score'] = self.create_social_support_score(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create device category features
        df = self.create_device_category_features(df)
        
        # Create demographic features
        df = self.create_demographic_features(df)
        
        # Create clinical features
        df = self.create_clinical_features(df)
        
        # Store composite scores
        self.composite_scores = {
            'digital_literacy_score': df['digital_literacy_score'],
            'technology_readiness_index': df['technology_readiness_index'],
            'health_risk_score': df['health_risk_score'],
            'social_support_score': df['social_support_score']
        }
        
        print(f"Created {len(df.columns)} features")
        
        return df
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of created features.
        """
        
        summary = {
            'composite_scores': list(self.composite_scores.keys()),
            'feature_importance': self.feature_importance,
            'total_features': len(self.composite_scores) + len(self.feature_importance)
        }
        
        return summary

def main():
    """Test the feature engineering module."""
    
    # Load sample data
    try:
        from src.data.generate_synthetic_data import AgeTechDataGenerator
        
        # Generate sample data
        generator = AgeTechDataGenerator(n_samples=100, random_state=42)
        df = generator.generate_dataset()
        
        # Initialize feature engineer
        engineer = AgeTechFeatureEngineer(random_state=42)
        
        # Create advanced features
        df_enhanced = engineer.create_advanced_features(df)
        
        print("\nFeature Engineering Summary:")
        print("=" * 40)
        print(f"Original features: {len(df.columns)}")
        print(f"Enhanced features: {len(df_enhanced.columns)}")
        print(f"New features created: {len(df_enhanced.columns) - len(df.columns)}")
        
        # Show composite scores
        print("\nComposite Scores Created:")
        for score_name in engineer.composite_scores.keys():
            print(f"  - {score_name}")
        
        # Show feature importance
        if engineer.feature_importance:
            print("\nTop 10 Most Important Features:")
            sorted_features = sorted(engineer.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error testing feature engineering: {e}")

if __name__ == "__main__":
    main() 