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
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Trained model artifacts
├── results/              # Model results and analysis
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Technical Implementation

- **Programming Environment**: Python with scikit-learn, XGBoost, NumPy, Pandas
- **Model Comparison**: Gradient Boosting (primary), Random Forest, Logistic Regression
- **Validation Framework**: 80-20 train-test split, 5-fold cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Interpretability**: SHAP analysis for feature importance

## Installation and Setup

### Basic Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run data preprocessing: `python src/data/preprocess.py`
4. Train models: `python src/models/train.py`
5. Evaluate performance: `python src/models/evaluate.py`

### Complete Pipeline Execution

#### Step 1: Run the Full Pipeline

Execute the complete ML workflow with one command:

```bash
python run_pipeline.py
```

This orchestrates:

1. **Synthetic Data Generation** - Creates 500-person dataset with realistic AgeTech adoption patterns
2. **Data Preprocessing** - Cleans and transforms the data for ML models
3. **Feature Engineering** - Creates advanced features and composite scores
4. **Model Training** - Trains multiple ML models (XGBoost, LightGBM, Random Forest, etc.)
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

   **Cell 5: Demographic Analysis**

   - Explores demographic factors and their relationship to adoption
   - **Visualizations**: 6 charts showing age groups, socioeconomic status, living situation distributions and adoption rates
   - **Key Insights**: Which demographic groups are most likely to adopt AgeTech

   **Cell 6: Technology Readiness Analysis**

   - Examines technology-related factors and their impact on adoption
   - **Visualizations**: 6 charts showing digital literacy, internet access, willingness to use new technology
   - **Key Insights**: How tech skills and attitudes affect adoption success

   **Cell 7: Correlation Analysis**

   - Shows relationships between all variables
   - **Visualizations**: Heatmap correlation matrix and top correlations list
   - **Key Insights**: Which factors most strongly predict AgeTech adoption

   **Alternative: Improved Correlation Matrix**

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

   **Example Correlation Heatmap:**

   <img width="810" height="673" alt="Example Correlation Heatmap" src="https://github.com/user-attachments/assets/ffb2a7e7-9976-438d-b8e2-d420217edb19" />

   **Cell 8: Key Findings Summary**

   - Synthesizes the most important predictive factors
   - **Visualizations**: Bar chart of top 5 predictors of AgeTech adoption success
   - **Output**: Summary of key findings and recommendations

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

### Understanding the Results

#### What the Visualizations Show:

- **Adoption Patterns**: How different groups adopt AgeTech
- **Predictive Factors**: Which variables most strongly influence adoption
- **Demographic Insights**: Age, socioeconomic, and living situation effects
- **Technology Barriers**: Digital literacy and tech readiness impacts
- **Correlation Strength**: Statistical relationships between variables

#### Key Metrics to Look For:

- **Overall Adoption Rate**: Typically 50-60% in synthetic data
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
