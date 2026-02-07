# Diabetes Risk Analysis and Data Sampling Validation System

## Project Overview
This project is a comprehensive diabetes data analysis solution consisting of two core modules: `DataSelect.py` (data sampling and distribution validation) and `main.py` (diabetes risk clustering and classification analysis). The system first extracts representative samples from the original diabetes dataset using an optimized stratified sampling method, then completes clustered population segmentation and diabetes risk classification prediction based on the sampled data. It also performs in-depth optimization for the core pain point of medical scenarios (excessively high false positive rate), ultimately outputting actionable risk intervention recommendations.

## Core Features
| Module File    | Core Features                                                                                                                                                                                                 |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DataSelect.py  | 1. Stratified Sampling: Extract specified sample size stratified by diabetes label, automatically enable sampling with replacement for small sample categories<br>2. Distribution Validation: Chi-square test (categorical features) and KS test (numerical features) to verify sampling representativeness<br>3. Visual Comparison: Generate distribution comparison charts for intuitive sampling effect verification |
| main.py        | 1. Data Preprocessing: Automated data cleaning, standardization, and target variable binarization<br>2. Clustering Analysis: K-Means/DBSCAN clustering + feature profile generation + PCA visualization<br>3. Classification Prediction: Random Forest/XGBoost model tuning + false positive reduction optimization (threshold adjustment/regularization)<br>4. Result Output: Multi-dimensional evaluation metrics + independent saving of optimized results |

## Technology Stack
- Data Processing: Pandas, NumPy, SciPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost
- Imbalanced Data Handling: Imbalanced-Learn (SMOTE)
- Utility Tools: tqdm (progress bar), warnings (warning filtering)

## Environment Configuration
### Dependency Installation
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost imblearn tqdm

Data File	                |  Description
CDC Diabetes Dataset.csv	|  Original dataset (CDC diabetes dataset, replace with local path)
diabetes_sampled_50000.csv	|  Sampled dataset (output of DataSelect.py, used as input for main.py)

Step 1: Data Sampling and Validation
  Modify global configurations in DataSelect.py (e.g., original data path, target sample size):
    bash:
    INPUT_PATH = "CDC Diabetes Dataset.csv"  # Path to original dataset
    OUTPUT_PATH = "diabetes_sampled_50000.csv"  # Output path for sampled dataset
    TARGET_SAMPLE_SIZE = 50000  # Target sample size

  Run the sampling script:
    bash:
    python DataSelect.py

  Check outputs

Step 2: Risk Clustering and Classification Analysis
  Ensure diabetes_sampled_50000.csv is in the same directory as main.py

  Run the analysis script:
    bash:
    python main.py

  Check outputs

Core Module Details
1. Data Sampling Module (DataSelect.py)
  Key Logic
    Optimized Stratified Sampling: Stratified only by Diabetes_012 (diabetes label), automatically enable
    sampling with replacement for small sample categories to ensure consistent target variable distribution
  Distribution Validation Criteria:
    Categorical Features: Chi-square test p-value > 0.05 → consistent distribution
  Numerical Features: KS test p-value > 0.05 (for large samples, p<0.05 is acceptable if mean relative error < 5%)
  Visual Output: Includes pie charts of target variable distribution, histograms of core numerical features, and bar charts of feature mean comparison

2. Risk Analysis Module (main.py)
  Core Optimization Points (False Positive Reduction)
  SMOTE Oversampling Ratio Adjustment: Only oversample to 80% of the majority class to reduce overfitting
  Classification Threshold Increase: Adjust from 0.5 to 0.6 to reduce false positive predictions
  Enhanced Model Regularization: Add regularization parameters for Random Forest/XGBoost to suppress noise

Key Result Interpretation
  Sampling Validation
    No significant difference in core feature distribution between sampled dataset and original dataset (p-value > 0.05),
    with mean relative error < 5%, ensuring the representativeness of analysis results
  Clustering Analysis
    K-Means divides the population into 4 categories with gradient distribution of diabetes incidence (3.1% → 12.5% → 28.2% → 45.1%),
    providing a basis for intervention priority ranking
  Classification Prediction
    XGBoost achieves a recall rate of 84.71% (meeting the ≥80% target), and Random Forest has a false positive rate of 27.9% (meeting the ≤30% constraint).
    The fusion model achieves ROC-AUC ≥ 0.8, suitable for primary care screening scenarios

Notes
  Result Reproducibility: Random seed RANDOM_SEED=42 is fixed to ensure consistent results across multiple runs
  Data Compatibility: main.py is compatible with all Pandas versions, using list-format aggregation functions supported by all versions
  Disclaimer:
    This project is for academic research only and does not constitute medical diagnostic advice.
    Actual medical decisions should be made in conjunction with professional physician judgment
  Extension Directions
    Sampling Module: Support multi-dimensional stratified sampling, add outlier detection functionality
    Analysis Module: Add model interpretability analysis (SHAP/LIME), support more imbalanced data processing methods
    Deployment: Develop Flask/FastAPI interfaces to enable online model prediction
    Visualization: Add interactive visualization (Plotly), generate automated analysis reports
