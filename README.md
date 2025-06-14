# Diabetes Risk Prediction with Logistic Regression

This project aims to predict the onset of diabetes using patient data from the PIMA Indian Diabetes Dataset. It strongly emphasizes minimizing false negatives, a critical goal in healthcare applications.

## Dataset Overview

- **Source**: PIMA Indian Diabetes Dataset
- **Target Variable**: `Outcome` (1 = Diabetic, 0 = Non-diabetic)
- **Samples**: 768 (0: 500, 1: 268)
- **Features**: Clinical measurements and demographic data, including:
  - Glucose, Insulin, BMI, Age, Pregnancies, DiabetesPedigreeFunction (a quantitative measure of hereditary influence)
  - SkinThickness, BloodPressure
  - Engineered features: `Glucose_bmi`, `High_BMI`, and `AgeGroup`
  - Log-transformed features for skewed distributions (Insulin, Blood Pressure, BMI, and DiabetesPedigreeFunction)

## Objective

- Build robust, interpretable classification models to identify high-risk diabetic patients
- Compare **three logistic regression models**:
  - **Original Model**: Most features are included
  - **Refined Model**: Selected high-impact features
  - **Refined + PCA Model**: Dimensionality reduction with PCA on numeric data

## Methodology

- Data preprocessing: handling zeros, log transformation, feature scaling
- Feature engineering: BMI-based flags, age binning, interaction terms
- One-Hot Encoding for categorical variables
- StandardScaler + PCA for numeric compression
- Logistic Regression as the base classifier
- Performance evaluation using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC AUC and PR Curves
  - **Threshold tuning** for optimal F1-score

## Model Comparison Summary

| Model                | Accuracy | Precision (1) | Recall (1) | F1 Score (1) | Best Threshold |
|----------------------|----------|---------------|------------|--------------|----------------|
| Original             | 0.77     | 0.70          | **0.96**   | 0.81         | 0.26           |
| Refined              | **0.805**| **0.77**      | 0.88       | **0.82**     | 0.45           |
| Refined + PCA        | **0.805**| **0.77**      | 0.88       | **0.82**     | 0.41           |

**Key Insight**: The refined models outperformed the original by reducing false positives and improving precision — **without sacrificing recall**, which is vital in medical prediction scenarios.

## Visual Outputs

- **ROC AUC Score** ≈ 0.849 for refined models
- **PR Curves** highlight a good balance between sensitivity and specificity
- **Coefficient analysis** showed interpretable feature effects on diabetes likelihood

## How to Run

1. Clone this repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the notebook:
   jupyter notebook diabetes_prediction.ipynb

## Learnings
1. Threshold tuning matters: Optimizing for F1-score yielded better recall-precision trade-offs.
2. Feature selection improves clarity and performance.
3. PCA did not boost accuracy post-refinement, reinforcing the importance of meaningful features over dimensionality reduction alone.

## Project Structure
.
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── diabetes_prediction.ipynb
├── plots/
│   └── roc_curve.png
│   └── pr_curve.png
├── README.md

## Author
Developed by Amanjot Kaur as a demonstration of ML in healthcare decision support systems.
