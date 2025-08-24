"""
Data Preprocessing Pipeline for AgeTech Adoption Prediction

This module handles data preprocessing including:
- Data validation and quality assessment
- Categorical variable encoding
- Feature engineering and scaling
- Train-test splitting with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class AgeTechPreprocessor:
    """
    Comprehensive preprocessing pipeline for AgeTech adoption data. Cleans and splits data before engineering features.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the synthetic dataset."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality and structure."""
        print("Validating data quality...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Warning: Missing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("✓ No missing values found")
        
        # Check data types
        print("\nData types:")
        print(df.dtypes)
        
        # Check for expected columns
        expected_columns = [
            'participant_id', 'age_group', 'socioeconomic_status', 'living_situation',
            'cognitive_status', 'physical_mobility', 'hearing_vision_impairment',
            'chronic_conditions', 'medication_effects', 'caregiver_support',
            'social_engagement', 'digital_literacy', 'internet_access',
            'attitude_toward_technology', 'previous_tech_use', 'agetch_experience',
            'tech_assistance_availability', 'willingness_new_tech', 'device_preferences',
            'adoption_success'
        ]
        
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            return False
        else:
            print("✓ All expected columns present")
        
        # Check outcome variable distribution
        adoption_rate = df['adoption_success'].mean()
        print(f"✓ Adoption rate: {adoption_rate:.2%}")
        
        if adoption_rate < 0.1 or adoption_rate > 0.9:
            print("Warning: Very imbalanced outcome variable")
        
        return True
    
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features."""
        
        # Exclude ID and outcome variables
        exclude_cols = ['participant_id', 'adoption_success']
        
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        
        return categorical_features, numerical_features
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced feature engineering."""
        print("Creating advanced features...")
        
        # Technology Readiness Composite Score
        tech_readiness_mapping = {
            'digital_literacy': {'Low': 1, 'Medium': 2, 'High': 3},
            'internet_access': {'None': 1, 'Limited': 2, 'Reliable': 3},
            'attitude_toward_technology': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
            'previous_tech_use': {'None': 1, 'Basic': 2, 'Intermediate': 3, 'Advanced': 4},
            'willingness_new_tech': {'Low': 1, 'Medium': 2, 'High': 3}
        }
        
        df['tech_readiness_score'] = 0
        for var, mapping in tech_readiness_mapping.items():
            df['tech_readiness_score'] += df[var].map(mapping)
        
        # Health Risk Composite Score
        health_risk_mapping = {
            'cognitive_status': {'No Impairment': 0, 'MCI': 1, 'Dementia': 2},
            'physical_mobility': {'Independent': 0, 'Assistive Device': 1, 'Full Assistance': 2},
            'hearing_vision_impairment': {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3},
            'chronic_conditions': {'0-1': 0, '2-3': 1, '4+': 2}
        }
        
        df['health_risk_score'] = 0
        for var, mapping in health_risk_mapping.items():
            df['health_risk_score'] += df[var].map(mapping)
        
        # Social Support Score
        social_support_mapping = {
            'caregiver_support': {'None': 0, 'Informal Only': 1, 'Formal Only': 1, 'Both': 2},
            'social_engagement': {'Isolated': 0, 'Moderate': 1, 'Active': 2},
            'tech_assistance_availability': {'None': 0, 'Limited': 1, 'Readily Available': 2}
        }
        
        df['social_support_score'] = 0
        for var, mapping in social_support_mapping.items():
            df['social_support_score'] += df[var].map(mapping)
        
        # Age Group Encoding
        age_mapping = {'65-74': 1, '75-84': 2, '85+': 3}
        df['age_group_encoded'] = df['age_group'].map(age_mapping)
        
        # Socioeconomic Status Encoding
        ses_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df['ses_encoded'] = df['socioeconomic_status'].map(ses_mapping)
        
        print("✓ Advanced features created")
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""
        
        # Remove ID column as it is not a feature
        if 'participant_id' in df.columns:
            df = df.drop('participant_id', axis=1)
        
        # Separate features and target
        X = df.drop('adoption_success', axis=1)
        y = df['adoption_success']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def clean_data_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data types to ensure categorical columns are strings."""
        
        X_clean = X.copy()
        
        # Convert all categorical columns to string type
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object' or col in self.categorical_features:
                # Fill NaN values with 'missing' and convert to string
                X_clean[col] = X_clean[col].fillna('missing').astype(str)
            elif X_clean[col].dtype in ['float64', 'int64']:
                # Fill NaN values with median for numerical columns
                median_val = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_val)
        
        return X_clean
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create sklearn preprocessing pipeline."""
        
        # Categorical preprocessing
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        # Numerical preprocessing
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine 
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.0) -> Tuple:
        """Split data into train and test sets (80-20 split with stratification)."""
        
        # Use 80-20 train-test split as per technical implementation
        print(f"Using full dataset: {X.shape[0]} samples")
        print(f"Class distribution: {y.value_counts(normalize=True)}")
        
        # Split into train and test (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )
        
        # Use same data for validation during training (for compatibility)
        X_val, y_val = X_train, y_train
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples (same as train for CV)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data_dict: Dict, output_dir: str = "data/processed"):
        """Save processed data to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            data.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")
    
    def save_preprocessor(self, preprocessor: Pipeline, output_dir: str = "models"):
        """Save the fitted preprocessor."""
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def preprocess_pipeline(self, input_filepath: str) -> Dict:
        """Complete preprocessing pipeline."""
        
        print("Starting AgeTech data preprocessing pipeline...")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_data(input_filepath)
        if df is None:
            return {}
        
        # 2. Validate data
        if not self.validate_data(df):
            print("Data validation failed!")
            return {}
        
        # 3. Create advanced features
        df = self.create_advanced_features(df)
        
        # 4. Identify feature types
        self.identify_feature_types(df)
        
        # 5. Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # 5.5. Clean data types before splitting
        X = self.clean_data_types(X)
        
        # 6. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 7. Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # 8. Fit preprocessor on training data
        print("\nFitting preprocessor...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert to dense arrays if they're sparse
        if hasattr(X_train_processed, 'toarray'):
            X_train_processed = X_train_processed.toarray()
        if hasattr(X_val_processed, 'toarray'):
            X_val_processed = X_val_processed.toarray()
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        
        # 9. Get feature names
        feature_names = []
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            # Ensure feature names match the actual output shape
            if len(feature_names) != X_train_processed.shape[1]:
                print(f"Warning: Feature names count ({len(feature_names)}) doesn't match output shape ({X_train_processed.shape[1]})")
                feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        else:
            # Create feature names based on actual output shape
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        print(f"Preprocessed data shape: {X_train_processed.shape}")
        print(f"Feature names count: {len(feature_names)}")
        
        # 10. Convert back to DataFrames
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        # 11. Add target variables
        X_train_df['adoption_success'] = y_train.values
        X_val_df['adoption_success'] = y_val.values
        X_test_df['adoption_success'] = y_test.values
        
        # 12. Save processed data
        processed_data = {
            'X_train': X_train_df,
            'X_val': X_val_df,
            'X_test': X_test_df,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        self.save_processed_data(processed_data)
        self.save_preprocessor(preprocessor)
        
        print("\nPreprocessing completed successfully!")
        print(f"Final feature count: {len(feature_names)}")
        
        return processed_data

def main():
    """Run the complete preprocessing pipeline."""
    
    # Find the most recent synthetic data file
    raw_data_dir = "data/raw"
    if not os.path.exists(raw_data_dir):
        print("No raw data directory found. Please generate synthetic data first.")
        return
    
    data_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not data_files:
        print("No CSV files found in raw data directory. Please generate synthetic data first.")
        return
    
    # Use the most recent synthetic data file (not demo)
    synthetic_files = [f for f in data_files if 'agetch_synthetic_data' in f and 'demo' not in f]
    if synthetic_files:
        latest_file = sorted(synthetic_files)[-1]
    else:
        latest_file = sorted(data_files)[-1]
    input_filepath = os.path.join(raw_data_dir, latest_file)
    
    print(f"Processing file: {input_filepath}")
    
    # Initialize preprocessor
    preprocessor = AgeTechPreprocessor(random_state=42)
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(input_filepath)
    
    if processed_data:
        print("\nPreprocessing Summary:")
        print("=" * 30)
        print(f"Training samples: {len(processed_data['X_train'])}")
        print(f"Validation samples: {len(processed_data['X_val'])}")
        print(f"Test samples: {len(processed_data['X_test'])}")
        print(f"Features: {processed_data['X_train'].shape[1] - 1}")  # Exclude target
        print(f"Adoption rate (train): {processed_data['y_train'].mean():.2%}")

if __name__ == "__main__":
    main() 