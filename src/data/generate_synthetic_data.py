"""
Synthetic Dataset Generator for AgeTech Adoption Prediction

This module generates a synthetic dataset of 500 individuals with realistic distributions
based on established research in gerontology, digital health, and technology adoption frameworks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AgeTechDataGenerator:
    """
    Generates synthetic AgeTech adoption data based on UTAUT framework and gerontology research.
    """
    
    def __init__(self, n_samples: int = 500, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        # Use consistent random seed for reproducible results
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define variable categories and their distributions
        self.demographic_vars = self._define_demographic_variables()
        self.health_cognitive_vars = self._define_health_cognitive_variables()
        self.social_support_vars = self._define_social_support_variables()
        self.tech_readiness_vars = self._define_tech_readiness_variables()
        
    def _define_demographic_variables(self) -> Dict:
        """Define demographic variables with realistic distributions."""
        return {
            'age_group': {
                'values': ['65-74', '75-84', '85+'],
                'probabilities': [0.45, 0.40, 0.15]  # Based on population demographics
            },
            'socioeconomic_status': {
                'values': ['Low', 'Medium', 'High'],
                'probabilities': [0.30, 0.50, 0.20]  # More realistic SES distribution
            },
            'living_situation': {
                'values': ['Independent Living', 'Assisted Living', 'Nursing Home', 'With Family'],
                'probabilities': [0.50, 0.20, 0.15, 0.15]  # More realistic distribution
            }
        }
    
    def _define_health_cognitive_variables(self) -> Dict:
        """Define health and cognitive variables based on clinical assessments."""
        return {
            'cognitive_status': {
                'values': ['No Impairment', 'MCI', 'Dementia'],
                'probabilities': [0.60, 0.30, 0.10]  # More realistic prevalence
            },
            'physical_mobility': {
                'values': ['Independent', 'Assistive Device', 'Full Assistance'],
                'probabilities': [0.60, 0.30, 0.10]
            },
            'hearing_vision_impairment': {
                'values': ['None', 'Mild', 'Moderate', 'Severe'],
                'probabilities': [0.40, 0.35, 0.20, 0.05]
            },
            'chronic_conditions': {
                'values': ['0-1', '2-3', '4+'],
                'probabilities': [0.30, 0.45, 0.25]
            },
            'medication_effects': {
                'values': ['None', 'Mild', 'Moderate', 'Significant'],
                'probabilities': [0.50, 0.30, 0.15, 0.05]
            }
        }
    
    def _define_social_support_variables(self) -> Dict:
        """Define social support variables."""
        return {
            'caregiver_support': {
                'values': ['None', 'Family', 'Professional', 'Both'],
                'probabilities': [0.20, 0.40, 0.25, 0.15]  # More realistic distribution
            },
            'social_engagement': {
                'values': ['Isolated', 'Moderate', 'Active'],
                'probabilities': [0.25, 0.50, 0.25]
            }
        }
    
    def _define_tech_readiness_variables(self) -> Dict:
        """Define technology readiness variables based on UTAUT framework."""
        return {
            'digital_literacy': {
                'values': ['Low', 'Medium', 'High'],
                'probabilities': [0.30, 0.45, 0.25]
            },
            'internet_access': {
                'values': ['None', 'Limited', 'Reliable'],
                'probabilities': [0.15, 0.35, 0.50]
            },
            'attitude_toward_technology': {
                'values': ['Negative', 'Neutral', 'Positive'],
                'probabilities': [0.20, 0.50, 0.30]
            },
            'previous_tech_use': {
                'values': ['None', 'Basic', 'Intermediate', 'Advanced'],
                'probabilities': [0.25, 0.40, 0.25, 0.10]
            },
            'agetch_experience': {
                'values': ['None', 'Limited', 'Moderate', 'Extensive'],
                'probabilities': [0.40, 0.35, 0.20, 0.05]
            },
            'tech_assistance_availability': {
                'values': ['None', 'Limited', 'Readily Available'],
                'probabilities': [0.20, 0.45, 0.35]
            },
            'willingness_new_tech': {
                'values': ['Low', 'Medium', 'High'],
                'probabilities': [0.25, 0.50, 0.25]
            },
            'device_preferences': {
                'values': ['Health Monitoring', 'Safety/Emergency', 'Communication', 'Cognitive Assistance', 'Mobility Assistance'],
                'probabilities': [0.25, 0.20, 0.20, 0.20, 0.15]
            }
        }
    
    def _generate_categorical_variable(self, var_def: Dict) -> List:
        """Generate categorical variable based on defined probabilities with random variation."""
        # Add small random variation to probabilities to make them truly random
        probs = np.array(var_def['probabilities'])
        noise = np.random.normal(0, 0.02, len(probs))
        probs = probs + noise
        probs = np.maximum(probs, 0.01)  # Ensure no negative probabilities
        probs = probs / probs.sum()  # Renormalize
        
        return np.random.choice(
            var_def['values'],
            size=self.n_samples,
            p=probs
        )
    
    def _create_interaction_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction effects between key variables."""
        
        # Digital literacy × Willingness interaction
        df['digital_willingness_interaction'] = (
            (df['digital_literacy'] == 'High').astype(int) * 
            (df['willingness_new_tech'] == 'High').astype(int)
        )
        
        # Cognitive × Tech assistance interaction
        df['cognitive_assistance_interaction'] = (
            (df['cognitive_status'] == 'No Impairment').astype(int) * 
            (df['tech_assistance_availability'] == 'Readily Available').astype(int)
        )
        
        # Social engagement × Caregiver support interaction
        df['social_caregiver_interaction'] = (
            (df['social_engagement'] == 'Active').astype(int) * 
            (df['caregiver_support'] != 'None').astype(int)
        )
        
        return df
    
    def _create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite scores for key domains."""
        
        # Technology Readiness Index
        tech_readiness_mapping = {
            'digital_literacy': {'Low': 1, 'Medium': 2, 'High': 3},
            'internet_access': {'None': 1, 'Limited': 2, 'Reliable': 3},
            'attitude_toward_technology': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
            'previous_tech_use': {'None': 1, 'Basic': 2, 'Intermediate': 3, 'Advanced': 4},
            'willingness_new_tech': {'Low': 1, 'Medium': 2, 'High': 3}
        }
        
        tech_scores = []
        for _, row in df.iterrows():
            score = 0
            for var, mapping in tech_readiness_mapping.items():
                score += mapping.get(row[var], 0)
            tech_scores.append(score)
        
        df['technology_readiness_index'] = tech_scores
        
        # Health Risk Score
        health_risk_mapping = {
            'cognitive_status': {'No Impairment': 0, 'MCI': 1, 'Dementia': 2},
            'physical_mobility': {'Independent': 0, 'Assistive Device': 1, 'Full Assistance': 2},
            'hearing_vision_impairment': {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3},
            'chronic_conditions': {'0-1': 0, '2-3': 1, '4+': 2}
        }
        
        health_scores = []
        for _, row in df.iterrows():
            score = 0
            for var, mapping in health_risk_mapping.items():
                score += mapping.get(row[var], 0)
            health_scores.append(score)
        
        df['health_risk_score'] = health_scores
        
        # Social Support Score
        social_support_mapping = {
            'caregiver_support': {'None': 0, 'Family': 1, 'Professional': 2, 'Both': 3},
            'social_engagement': {'Isolated': 0, 'Moderate': 1, 'Active': 2}
        }
        
        social_scores = []
        for _, row in df.iterrows():
            score = 0
            for var, mapping in social_support_mapping.items():
                score += mapping.get(row[var], 0)
            social_scores.append(score)
        
        df['social_support_score'] = social_scores
        
        # Digital Literacy Score (numeric)
        digital_literacy_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['digital_literacy_score'] = df['digital_literacy'].map(digital_literacy_mapping)
        
        # Technology Willingness Score (numeric)
        tech_willingness_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['technology_willingness'] = df['willingness_new_tech'].map(tech_willingness_mapping)
        
        return df
    
    def _generate_adoption_outcome(self, df: pd.DataFrame) -> List[int]:
        """Generate adoption outcome based on key predictors with varied adoption rates."""
        
        # Create truly random base adoption probabilities
        base_rates = np.random.uniform(0.30, 0.80, self.n_samples)
        
        adoption_probs = []
        for i, (_, row) in enumerate(df.iterrows()):
            base_prob = base_rates[i]
            
            # Age affects adoption rate
            if row['age_group'] == '65-74':
                age_modifier = 0.15
            elif row['age_group'] == '75-84':
                age_modifier = 0.0
            else:  # 85+
                age_modifier = -0.2
            
            # Socioeconomic status affects adoption
            if row['socioeconomic_status'] == 'High':
                ses_modifier = 0.2
            elif row['socioeconomic_status'] == 'Medium':
                ses_modifier = 0.05
            else:  # Low
                ses_modifier = -0.15
            
            # Digital literacy strongly affects adoption
            if row['digital_literacy'] == 'High':
                digital_modifier = 0.25
            elif row['digital_literacy'] == 'Medium':
                digital_modifier = 0.05
            else:  # Low
                digital_modifier = -0.15
            
            # Technology willingness affects adoption
            if row['willingness_new_tech'] == 'High':
                tech_modifier = 0.20
            elif row['willingness_new_tech'] == 'Medium':
                tech_modifier = 0.05
            else:  # Low
                tech_modifier = -0.20
            
            # Cognitive status affects adoption
            if row['cognitive_status'] == 'No Impairment':
                cognitive_modifier = 0.15
            elif row['cognitive_status'] == 'MCI':
                cognitive_modifier = -0.05
            else:  # Dementia
                cognitive_modifier = -0.25
            
            # Caregiver support affects adoption
            if row['caregiver_support'] != 'None':
                caregiver_modifier = 0.10
            else:
                caregiver_modifier = 0.0
            
            # Tech assistance availability affects adoption
            if row['tech_assistance_availability'] == 'Readily Available':
                assistance_modifier = 0.15
            elif row['tech_assistance_availability'] == 'Limited':
                assistance_modifier = 0.05
            else:  # None
                assistance_modifier = -0.10
            
            # Attitude toward technology affects adoption
            if row['attitude_toward_technology'] == 'Positive':
                attitude_modifier = 0.15
            elif row['attitude_toward_technology'] == 'Neutral':
                attitude_modifier = 0.0
            else:  # Negative
                attitude_modifier = -0.20
            
            # Calculate final adoption probability
            adoption_prob = (base_prob + age_modifier + ses_modifier + digital_modifier + 
                           tech_modifier + cognitive_modifier + caregiver_modifier + 
                           assistance_modifier + attitude_modifier)
            
            # Add more random noise to prevent patterns
            adoption_prob += np.random.normal(0, 0.10)
            
            # Clip to valid probability range
            adoption_prob = max(0.05, min(0.95, adoption_prob))
            
            adoption_probs.append(adoption_prob)
        
        # Generate outcomes
        outcomes = np.random.binomial(1, adoption_probs)
        return outcomes.tolist()
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset."""
        
        # Initialize empty dataframe
        data = {}
        
        # Generate demographic variables
        for var_name, var_def in self.demographic_vars.items():
            data[var_name] = self._generate_categorical_variable(var_def)
        
        # Generate health/cognitive variables
        for var_name, var_def in self.health_cognitive_vars.items():
            data[var_name] = self._generate_categorical_variable(var_def)
        
        # Generate social support variables
        for var_name, var_def in self.social_support_vars.items():
            data[var_name] = self._generate_categorical_variable(var_def)
        
        # Generate technology readiness variables
        for var_name, var_def in self.tech_readiness_vars.items():
            data[var_name] = self._generate_categorical_variable(var_def)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Add interaction effects
        df = self._create_interaction_effects(df)
        
        # Add composite scores
        df = self._create_composite_scores(df)
        
        # Generate adoption outcome
        df['adoption_success'] = self._generate_adoption_outcome(df)
        
        # Add participant ID
        df.insert(0, 'participant_id', range(1, len(df) + 1))
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save the generated dataset to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agetch_synthetic_data_{timestamp}.csv"
        
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Adoption rate: {df['adoption_success'].mean():.2%}")
        
        return filepath

def main():
    """Generate and save the synthetic dataset."""
    print("Generating AgeTech synthetic dataset...")
    
    # Initialize generator
    generator = AgeTechDataGenerator(n_samples=500, random_state=42)
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Save dataset
    filepath = generator.save_dataset(df)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("=" * 50)
    print(f"Total participants: {len(df)}")
    print(f"Adoption success rate: {df['adoption_success'].mean():.2%}")
    print(f"Number of features: {len(df.columns) - 2}")  # Exclude ID and outcome
    
    print("\nKey Variable Distributions:")
    print("-" * 30)
    key_vars = ['age_group', 'cognitive_status', 'digital_literacy', 'willingness_new_tech']
    for var in key_vars:
        print(f"{var}:")
        print(df[var].value_counts(normalize=True).round(3))
        print()
    
    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    main() 