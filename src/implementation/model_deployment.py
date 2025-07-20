"""
Model Deployment Module for AgeTech Adoption Prediction

This module provides production deployment capabilities including:
- Model serving and API development
- Docker containerization
- Production monitoring and logging
- Model versioning and updates
- Performance tracking
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AgeTechModelDeployment:
    """
    Production deployment framework for AgeTech adoption prediction models.
    """
    
    def __init__(self, model_dir: str = "models", config_file: str = "config.yaml"):
        self.model_dir = model_dir
        self.config_file = config_file
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_metadata = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for deployment monitoring."""
        
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/deployment.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_production_model(self) -> bool:
        """
        Load the best model for production deployment.
        """
        
        try:
            # Find best model
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('best_model_')]
            if not model_files:
                self.logger.error("No best model found!")
                return False
            
            best_model_file = model_files[0]
            model_path = os.path.join(self.model_dir, best_model_file)
            
            # Load model
            self.model = joblib.load(model_path)
            self.logger.info(f"Loaded production model: {best_model_file}")
            
            # Load preprocessor if available
            preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                self.logger.info("Loaded preprocessor")
            
            # Load feature engineer
            try:
                from src.features.engineering import AgeTechFeatureEngineer
                self.feature_engineer = AgeTechFeatureEngineer()
                self.logger.info("Initialized feature engineer")
            except Exception as e:
                self.logger.warning(f"Could not load feature engineer: {e}")
            
            # Load model metadata
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                self.logger.info("Loaded model metadata")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading production model: {e}")
            return False
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        """
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Apply feature engineering if available
            if self.feature_engineer:
                df = self.feature_engineer.create_advanced_features(df)
            
            # Apply preprocessor if available
            if self.preprocessor:
                df_processed = self.preprocessor.transform(df)
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names = self.preprocessor.get_feature_names_out()
                else:
                    feature_names = [f"feature_{i}" for i in range(df_processed.shape[1])]
                
                df = pd.DataFrame(df_processed, columns=feature_names)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing input: {e}")
            raise
    
    def predict(self, input_data: Dict) -> Dict:
        """
        Make prediction for a single input.
        """
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            # Create response
            response = {
                'prediction': int(prediction),
                'probability': float(probability),
                'adoption_likelihood': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low',
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('version', 'unknown')
            }
            
            # Log prediction
            self.logger.info(f"Prediction made: {prediction}, Probability: {probability:.3f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def batch_predict(self, input_data_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple inputs.
        """
        
        try:
            results = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    result = self.predict(input_data)
                    result['input_index'] = i
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error predicting input {i}: {e}")
                    results.append({
                        'error': str(e),
                        'input_index': i,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get model information and metadata.
        """
        
        info = {
            'model_type': type(self.model).__name__,
            'model_version': self.model_metadata.get('version', 'unknown'),
            'training_date': self.model_metadata.get('training_date', 'unknown'),
            'performance_metrics': self.model_metadata.get('performance_metrics', {}),
            'feature_count': self.model_metadata.get('feature_count', 'unknown'),
            'model_size_mb': self.get_model_size(),
            'last_updated': datetime.now().isoformat()
        }
        
        return info
    
    def get_model_size(self) -> float:
        """Get model file size in MB."""
        
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('best_model_')]
            if model_files:
                model_path = os.path.join(self.model_dir, model_files[0])
                size_bytes = os.path.getsize(model_path)
                return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            pass
        
        return 0.0
    
    def create_api_spec(self) -> Dict:
        """
        Create OpenAPI specification for the prediction API.
        """
        
        api_spec = {
            'openapi': '3.0.0',
            'info': {
                'title': 'AgeTech Adoption Prediction API',
                'description': 'API for predicting AgeTech adoption success in older adults',
                'version': '1.0.0'
            },
            'paths': {
                '/predict': {
                    'post': {
                        'summary': 'Predict AgeTech adoption success',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'age_group': {'type': 'string', 'enum': ['65-74', '75-84', '85+']},
                                            'socioeconomic_status': {'type': 'string', 'enum': ['Low', 'Medium', 'High']},
                                            'living_situation': {'type': 'string'},
                                            'cognitive_status': {'type': 'string', 'enum': ['No Impairment', 'MCI', 'Dementia']},
                                            'physical_mobility': {'type': 'string'},
                                            'digital_literacy': {'type': 'string', 'enum': ['Basic', 'Intermediate', 'Advanced']},
                                            'attitude_toward_technology': {'type': 'string', 'enum': ['Negative', 'Neutral', 'Positive']},
                                            'willingness_to_use_new_technology': {'type': 'string', 'enum': ['Unwilling', 'Neutral', 'Willing']}
                                        },
                                        'required': ['age_group', 'socioeconomic_status', 'cognitive_status', 'digital_literacy']
                                    }
                                }
                            }
                        },
                        'responses': {
                            '200': {
                                'description': 'Successful prediction',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'prediction': {'type': 'integer', 'enum': [0, 1]},
                                                'probability': {'type': 'number'},
                                                'adoption_likelihood': {'type': 'string'},
                                                'timestamp': {'type': 'string'},
                                                'model_version': {'type': 'string'}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                '/batch-predict': {
                    'post': {
                        'summary': 'Batch predict AgeTech adoption success',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'inputs': {
                                                'type': 'array',
                                                'items': {'$ref': '#/components/schemas/PredictionInput'}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        'responses': {
                            '200': {
                                'description': 'Successful batch prediction',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'predictions': {
                                                    'type': 'array',
                                                    'items': {'$ref': '#/components/schemas/PredictionOutput'}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                '/model-info': {
                    'get': {
                        'summary': 'Get model information',
                        'responses': {
                            '200': {
                                'description': 'Model information',
                                'content': {
                                    'application/json': {
                                        'schema': {'$ref': '#/components/schemas/ModelInfo'}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'components': {
                'schemas': {
                    'PredictionInput': {
                        'type': 'object',
                        'properties': {
                            'age_group': {'type': 'string'},
                            'socioeconomic_status': {'type': 'string'},
                            'cognitive_status': {'type': 'string'},
                            'digital_literacy': {'type': 'string'}
                        }
                    },
                    'PredictionOutput': {
                        'type': 'object',
                        'properties': {
                            'prediction': {'type': 'integer'},
                            'probability': {'type': 'number'},
                            'adoption_likelihood': {'type': 'string'},
                            'timestamp': {'type': 'string'}
                        }
                    },
                    'ModelInfo': {
                        'type': 'object',
                        'properties': {
                            'model_type': {'type': 'string'},
                            'model_version': {'type': 'string'},
                            'training_date': {'type': 'string'},
                            'performance_metrics': {'type': 'object'}
                        }
                    }
                }
            }
        }
        
        return api_spec
    
    def create_dockerfile(self) -> str:
        """
        Create Dockerfile for containerization.
        """
        
        dockerfile = """# AgeTech Adoption Prediction API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

# Run the application
CMD ["python", "-m", "src.implementation.api_server"]
"""
        
        return dockerfile
    
    def create_docker_compose(self) -> str:
        """
        Create docker-compose.yml for easy deployment.
        """
        
        compose = """version: '3.8'

services:
  agetch-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - MODEL_DIR=/app/models
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
"""
        
        return compose
    
    def create_health_check(self) -> Dict:
        """
        Create health check endpoint response.
        """
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
            'feature_engineer_loaded': self.feature_engineer is not None,
            'model_info': self.get_model_info() if self.model else None
        }
        
        return health_status
    
    def save_deployment_files(self, output_dir: str = "deployment"):
        """
        Save all deployment files.
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save API specification
        api_spec = self.create_api_spec()
        with open(os.path.join(output_dir, "api_spec.json"), 'w') as f:
            json.dump(api_spec, f, indent=2)
        
        # Save Dockerfile
        dockerfile = self.create_dockerfile()
        with open(os.path.join(output_dir, "Dockerfile"), 'w') as f:
            f.write(dockerfile)
        
        # Save docker-compose
        compose = self.create_docker_compose()
        with open(os.path.join(output_dir, "docker-compose.yml"), 'w') as f:
            f.write(compose)
        
        # Save deployment configuration
        deployment_config = {
            'model_dir': self.model_dir,
            'api_port': 8000,
            'log_level': 'INFO',
            'max_batch_size': 100,
            'timeout_seconds': 30
        }
        
        with open(os.path.join(output_dir, "deployment_config.json"), 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        self.logger.info(f"Deployment files saved to {output_dir}")
    
    def run_deployment_test(self) -> Dict:
        """
        Run deployment test with sample data.
        """
        
        try:
            # Sample test data
            test_input = {
                'age_group': '75-84',
                'socioeconomic_status': 'Medium',
                'living_situation': 'Independent Living',
                'cognitive_status': 'No Impairment',
                'physical_mobility': 'Independent',
                'hearing_vision_impairment': 'Mild',
                'chronic_conditions': '1-2',
                'medication_effects': '1-2',
                'caregiver_support_availability': 'Available',
                'social_engagement_level': 'Moderate',
                'digital_literacy': 'Intermediate',
                'internet_access': 'Reliable',
                'attitude_toward_technology': 'Positive',
                'previous_tech_use': 'Moderate',
                'agetch_experience': 'Some',
                'tech_assistance_availability': 'Available',
                'willingness_to_use_new_technology': 'Willing',
                'device_preferences': 'Smart Watch, Health Tracker'
            }
            
            # Test single prediction
            single_result = self.predict(test_input)
            
            # Test batch prediction
            batch_result = self.batch_predict([test_input, test_input])
            
            # Test health check
            health_status = self.create_health_check()
            
            test_results = {
                'single_prediction': single_result,
                'batch_prediction': batch_result,
                'health_check': health_status,
                'model_info': self.get_model_info(),
                'test_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Deployment test completed successfully")
            return test_results
            
        except Exception as e:
            self.logger.error(f"Deployment test failed: {e}")
            return {'error': str(e)}
    
    def deploy(self) -> bool:
        """
        Complete deployment process.
        """
        
        self.logger.info("Starting AgeTech model deployment...")
        
        # Load model
        if not self.load_production_model():
            self.logger.error("Failed to load production model")
            return False
        
        # Run deployment test
        test_results = self.run_deployment_test()
        if 'error' in test_results:
            self.logger.error(f"Deployment test failed: {test_results['error']}")
            return False
        
        # Save deployment files
        self.save_deployment_files()
        
        self.logger.info("Deployment completed successfully")
        return True

def main():
    """Run the deployment process."""
    
    # Initialize deployment
    deployment = AgeTechModelDeployment()
    
    # Run deployment
    success = deployment.deploy()
    
    if success:
        print("\nDeployment Summary:")
        print("=" * 30)
        print("✓ Model loaded successfully")
        print("✓ Deployment test passed")
        print("✓ Deployment files created")
        print("✓ Ready for production")
        
        # Show model info
        model_info = deployment.get_model_info()
        print(f"\nModel Information:")
        print(f"• Type: {model_info['model_type']}")
        print(f"• Version: {model_info['model_version']}")
        print(f"• Size: {model_info['model_size_mb']:.2f} MB")
        
        # Show deployment files
        print(f"\nDeployment files created in 'deployment/' directory:")
        print("• api_spec.json - OpenAPI specification")
        print("• Dockerfile - Container configuration")
        print("• docker-compose.yml - Multi-service deployment")
        print("• deployment_config.json - Deployment settings")
        
    else:
        print("Deployment failed. Check logs for details.")

if __name__ == "__main__":
    main() 