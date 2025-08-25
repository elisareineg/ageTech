# AgeTech Adoption Prediction Model

## Project Overview

This project develops a machine learning model to predict AgeTech adoption success in real-world clinical populations of older adults. The model identifies key factors influencing technology adoption and provides insights for healthcare providers and policymakers.

## Research Questions

- **Primary**: Can a machine learning model accurately predict AgeTech adoption success in real-world clinical populations of older adults, and what factors most strongly influence these predictions?
- **Secondary**:
  - How does model performance vary across demographic and clinical subgroups?
  - What predictive factors emerge in real-world data not captured in synthetic datasets?
  - How do prediction accuracy rates compare between different AgeTech device categories?

## Dataset Structure

The synthetic dataset includes 500 individuals with 18 variables across four domains:

### Demographic Variables (3)

- Age group
- Socioeconomic status
- Living situation

### Health/Cognitive Variables (5)

- Cognitive status (No Impairment/MCI/Dementia)
- Physical mobility
- Hearing/vision impairment
- Chronic conditions
- Medication effects

### Social Support Variables (2)

- Caregiver support availability
- Social engagement level

### Technology Readiness Variables (8)

- Digital literacy
- Internet access
- Attitude toward technology
- Previous tech use
- AgeTech experience
- Tech assistance availability
- Willingness to use new technology
- Device preferences

### Outcome Variable

- Binary AgeTech adoption success

## Project Structure

```
ageTech/
├── data/                   # Data files
│   ├── raw/               # Original synthetic dataset
│   ├── processed/         # Preprocessed data
│   └── external/          # External validation data
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Plotting and analysis
├── visualizations/       # Generated charts and visualizations
├── models/               # Trained model artifacts
├── results/              # Model results and analysis
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Technical Implementation

- **Programming Environment**: Python with scikit-learn, NumPy, Pandas
- **Model Comparison**: Gradient Boosting (primary), Random Forest, Logistic Regression
- **Data Configuration**: Full 500 samples used for training and validation (80% train, 20% validation)
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Target Performance**: Precision >90%, Recall >85%
- **Interpretability**: SHAP analysis for feature importance (when available)
- **Visualization**: Direct chart generation (PNG files) 

### **Visualization System**

`src/visualization/generate_charts.py` - Direct chart generation system

- **Features**: 11 comprehensive visualizations (PNG files), 
- **Output**: High-quality charts for data overview, model performance, feature importance, correlations
- **Usage**: Automatic (via pipeline) or manual: `python src/visualization/generate_charts.py`
- **Result**: Professional visualizations saved in `visualizations/` directory
- **File Management**: Uses "latest" filenames for current results, auto-cleanup of old files through `python src/visualization/cleanup_old_files.py` which is intergrated into the pipeline.

## Installation and Setup

### Basic Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the complete pipeline: `python run_pipeline.py`

### Complete Pipeline Execution

#### Step 1: Run the Full Pipeline

Execute the complete ML workflow with one command:

```bash
python run_pipeline.py
```

This orchestrates:

1. **Synthetic Data Generation** - Creates 500-person dataset with realistic AgeTech adoption patterns
2. **Data Preprocessing** - Cleans and transforms the data for ML models (uses full 500 samples)
3. **Feature Engineering** - Creates advanced features and composite scores
4. **Model Training** - Trains multiple ML models (Gradient Boosting, Random Forest, Logistic Regression)
5. **Model Evaluation** - Assesses performance across different metrics and subgroups
6. **Model Interpretability** - Generates SHAP analysis for feature importance
   

**For a quick overview of results, run:**

```bash
python run_analysis.py
```

This provides:

- Overall adoption rate
- Highest adoption rates by demographic groups
- Key predictive factors
- Dataset quality metrics

## Important Configuration Notes

### Data Splitting Strategy

- **Current Setup**: Uses full 500 samples for model training and validation
- **Split**: 80% training (400 samples), 20% validation (100 samples)
- **No Separate Test Set**: Validation performance represents final model performance
- **Rationale**: Maximizes data usage for model training while maintaining evaluation capability

### Model Configuration

- **Primary Models**: Gradient Boosting, Random Forest, Logistic Regression
- **Feature Selection**: Top 15 features selected using F-statistic
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Class Balancing**: Uses `class_weight='balanced'` for imbalanced datasets

### Pipeline Behavior

- **Graceful Degradation**: Optional dependencies (XGBoost, LightGBM, SHAP) are handled gracefully
- **Error Handling**: Pipeline continues even if individual modules encounter issues
- **Comprehensive Logging**: Detailed progress and error reporting

### Understanding the Results

#### What the Visualizations Show:

- **Adoption Patterns**: How different groups adopt AgeTech
- **Predictive Factors**: Which variables most strongly influence adoption
- **Demographic Insights**: Age, socioeconomic, and living situation effects
- **Technology Barriers**: Digital literacy and tech readiness impacts
- **Correlation Strength**: Statistical relationships between variables

#### Key Metrics to Look For & Understand:

**AgeTech Adoption Rate (25-45%)**

- **Expected Range**: 25-45% overall adoption rate
- **Why This Range**: Based on real-world research showing that technology adoption among older adults (65+) typically falls between 25-45%. This is based off studies on internet and smartphone use, as well as other variables listed in this study.
- **Realistic Variation**: Each pipeline run generates slightly different adoption rates within this range due to the probabilistic nature of the synthetic data generation

**🎲 Adoption Probability Breakdown**

The synthetic data generation uses a multi-factor probability model:

**Base Rate (20-50%)**: Starting probability for each individual

**Age Modifiers**:

- 65-74: +15% (highest adoption)
- 75-84: -10% (moderate decline)
- 85+: -25% (significant decline)

**Digital Literacy Modifiers**:

- High: +25% (strong positive)
- Medium: 0% (neutral)
- Low: -25% (strong negative)

**Technology Willingness Modifiers**:

- High: +20% (strong positive)
- Medium: 0% (neutral)
- Low: -20% (strong negative)

**Cognitive Status Modifiers**:

- No Impairment: +10% (positive)
- MCI: -10% (moderate negative)
- Dementia: -25% (strong negative)

**Additional Factors**: Socioeconomic status, caregiver support, tech assistance availability, attitude toward technology, physical mobility, and sensory impairments all contribute smaller but significant modifiers.

**Final Probability**: All modifiers are combined with controlled random noise (±8%) and clipped to realistic bounds (5-75%), ensuring adoption rates stay within the 25-45% target range while maintaining realistic individual variation.

## Ethical Considerations

- Fairness metrics across demographic subgroups
- Bias detection and mitigation
- Transparency in decision-making processes
- Equitable access to AgeTech recommendations

## Future Work

- Real-world validation study (Phase 2)
- Clinical decision support tool development
- Economic evaluation and cost-effectiveness analysis
- Model expansion with additional predictive variables

### Model Deployment (Production)

When ready to deploy the trained model to production, use the deployment module:

```bash
python src/implementation/model_deployment.py
```

This creates production-ready deployment files:

- **API Specification** (`api_spec.json`) - OpenAPI documentation for REST endpoints
- **Dockerfile** - Container configuration for deployment
- **Docker Compose** (`docker-compose.yml`) - Multi-service deployment setup
- **Deployment Config** (`deployment_config.json`) - Production settings

**Note**: Run the main pipeline first to ensure trained models exist before deployment.
