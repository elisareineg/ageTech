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

## Synthetic Data Generation Mechanism

### Adoption Rate Design

The synthetic dataset consistently shows adoption rates around **50-60%** due to the realistic probability-based generation algorithm:

#### **Base Probability: 35%**

The algorithm starts with a 35% base adoption probability, which reflects real-world AgeTech adoption research.

#### **Individual Probability Adjustments**

Each person's adoption probability is adjusted based on their characteristics:

**Positive Factors (Increase Probability):**

- **High digital literacy**: +25% probability boost
- **High willingness to use new technology**: +20% probability boost
- **No cognitive impairment**: +15% probability boost
- **Available caregiver support**: +10% probability boost
- **Readily available tech assistance**: +15% probability boost
- **Positive attitude toward technology**: +15% probability boost
- **Digital-willingness interaction**: +10% probability boost

**Negative Factors (Decrease Probability):**

- **Low digital literacy**: -15% probability reduction
- **Low willingness to use new technology**: -20% probability reduction
- **Dementia**: -25% probability reduction
- **Negative attitude toward technology**: -20% probability reduction

#### **Probability Bounds**

Final probabilities are constrained between 5% and 95% to ensure realistic outcomes.

#### **Why 50-60% Overall Rate?**

1. **Balanced Population**: The synthetic data creates a realistic mix of people with varying characteristics
2. **Multiple Positive Factors**: Many individuals have several positive factors that boost their adoption probability
3. **Research-Based Design**: The 35% base rate and adjustment factors are based on real-world AgeTech adoption studies
4. **Natural Variation**: Random sampling from individual probabilities creates realistic distribution

**Example Calculations:**

- **High-adoption individual**: 35% base + 25% (high literacy) + 20% (high willingness) + 15% (no impairment) = 95% probability
- **Low-adoption individual**: 35% base - 15% (low literacy) - 20% (low willingness) - 25% (dementia) = 5% probability (minimum bound)

This design ensures the synthetic data reflects realistic AgeTech adoption patterns where adoption is moderate but achievable, with clear predictive factors that the ML model can learn to identify.

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
├── notebooks/            # Jupyter notebooks for exploration
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

## Current Model Performance

**Best Model: Gradient Boosting**

- **Accuracy**: 96.2%
- **Precision**: 95.4%
- **Recall**: 98.0%
- **F1 Score**: 96.7%

**Top Predictive Features:**

1. Technology Readiness Index
2. Willingness to use new technology
3. Cognitive status
4. Health risk score
5. Previous technology use

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
7. **Data Visualization** - Prepares Jupyter notebook for exploration

#### Step 2: View Data Visualizations in Jupyter Notebook

1. **Start Jupyter Server**

   ```bash
   jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
   ```

2. **Access the Notebook**

   - Copy the URL from terminal output (looks like: `http://127.0.0.1:8888/tree?token=...`)
   - Navigate to the `notebooks` folder
   - Click on `01_data_exploration.ipynb`
   - If prompted, click "Trust" to allow execution

3. **Run Visualization Cells**
   The notebook contains 8 cells that generate comprehensive visualizations:

   **Cell 1: Library Import**

   - Imports pandas, numpy, matplotlib, seaborn
   - Sets up plotting styles and configurations
   - **Output**: "Libraries imported successfully!"

   **Cell 2: Data Loading**

   - Loads the synthetic dataset from `data/raw/agetch_synthetic_data_[TIMESTAMP].csv`
   - **Output**: Dataset shape, column names, confirmation of successful loading

   **Cell 3: Dataset Overview**

   - Shows basic dataset information
   - **Visualizations**: Data types summary, missing values count, first 5 rows of data
   - **Output**: Shape, data types, missing values, sample data

   **Cell 4: Target Variable Analysis**

   - Analyzes the binary adoption success outcome
   - **Visualizations**: Pie chart and bar chart of adoption success distribution
   - **Output**: Adoption rate percentage and distribution statistics

   **Target Variable Analysis Based Off Sample Dataset:**

   <img width="909" height="427" alt="Screenshot 2025-08-23 at 4 27 35 PM" src="https://github.com/user-attachments/assets/09ade9ad-5054-4ad6-8715-d0764cb8e698" />

   **Cell 5: Demographic Analysis**

   - Explores demographic factors and their relationship to adoption
   - **Visualizations**: 6 charts showing age groups, socioeconomic status, living situation distributions and adoption rates
   - **Key Insights**: Which demographic groups are most likely to adopt AgeTech

   **Demographic Analysis Based Off Sample Dataset**

   <img width="942" height="594" alt="Screenshot 2025-08-23 at 4 28 15 PM" src="https://github.com/user-attachments/assets/78a983e8-69cd-493e-9b5a-8e3c1ee71b76" />

   **Cell 6: Technology Readiness Analysis**

   - Examines technology-related factors and their impact on adoption
   - **Visualizations**: 6 charts showing digital literacy, internet access, willingness to use new technology
   - **Key Insights**: How tech skills and attitudes affect adoption success

   **Technology Readiness Analysis Based Off Sample Dataset:**

   <img width="900" height="626" alt="Screenshot 2025-08-23 at 4 29 00 PM" src="https://github.com/user-attachments/assets/1fd37744-8ef7-4f2f-8e95-35923fd3f7f9" />

   **Cell 7: Correlation Analysis**

   - Shows relationships between all variables
   - **Visualizations**: Heatmap correlation matrix and top correlations list
   - **Key Insights**: Which factors most strongly predict AgeTech adoption

   **Example Correlation Heatmap:**

   <img width="810" height="673" alt="Example Correlation Heatmap" src="https://github.com/user-attachments/assets/ffb2a7e7-9976-438d-b8e2-d420217edb19" />

   **Alternative: Focused Correlation Matrix**

   For a more organized and readable correlation matrix, run:

   ```bash
   python better_correlation.py
   ```

   This provides:

   - **Focused heatmap**: Shows only the top 8 most important predictors
   - **Larger text**: Easier to read correlation values
   - **Better colors**: Clearer contrast for positive/negative correlations
   - **Clean table**: Top 10 correlations clearly ranked and listed
   - **Key insights**: Focuses on the relationships that matter most

   **Example Focused Heatmap:**

   <img width="1204" height="842" alt="Screenshot 2025-08-23 at 4 21 54 PM" src="https://github.com/user-attachments/assets/ac8acc89-ea49-41ce-8767-aa922da5948b" />

   **Cell 8: Key Findings Summary**

   - Synthesizes the most important predictive factors
   - **Visualizations**: Bar chart of top 5 predictors of AgeTech adoption success
   - **Output**: Summary of key findings and recommendations

   **Key Findings Summary Based Off Sample Dataset:**

   <img width="946" height="640" alt="Screenshot 2025-08-23 at 4 29 39 PM" src="https://github.com/user-attachments/assets/3b4b358b-ffc0-4b6a-92eb-f5631cf02d4f" />

### Alternative: Quick Data Analysis

For a quick overview without Jupyter, run:

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

#### Key Metrics to Look For:

- **Overall Adoption Rate**: Typically 50-60% in synthetic data

  **Why 50-60% Adoption Rate?**

  The consistent 50-60% adoption rate is by design, reflecting realistic AgeTech adoption patterns:

  - **Base Probability**: 35% starting point based on AgeTech research
  - **Individual Adjustments**: Each person's probability is modified by their characteristics:
    - **Positive factors** (high digital literacy, willingness, no cognitive impairment) increase probability
    - **Negative factors** (low literacy, dementia, negative attitudes) decrease probability
  - **Probability Bounds**: Constrained between 5% and 95% for realistic outcomes
  - **Balanced Population**: Creates realistic mix of high and low adoption individuals
  - **Research-Based**: Reflects actual AgeTech adoption studies in clinical populations

- **Top Predictors**: Usually digital literacy, willingness to use tech, cognitive status
- **Demographic Variations**: Different adoption rates across age groups and socioeconomic levels
- **Technology Readiness**: Clear relationship between tech skills and adoption success

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
