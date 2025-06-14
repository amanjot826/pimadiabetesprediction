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

| Metric              | Original (Threshold = 0.35)  | Refined (Threshold = 0.45)  | Refined + PCA (Threshold = 0.45) |
|---------------------|------------------------------|-----------------------------|----------------------------------|
| **Accuracy**        | 0.79                         | **0.805**                   | **0.805**                        |
| **Precision (1)**   | 0.73                         | **0.77**                    | **0.77**                         |
| **Recall (1)**      | **0.91**                     | 0.88                        | 0.88                             |
| **F1 Score (1)**    | 0.81                         | **0.82**                    | **0.82**                         |
| **TN / FP**         | 67 / 33                      | **73 / 27**                 | **73 / 27**                      |
| **FN / TP**         | **9 / 91**                   | 12 / 88                     | 12 / 88                          |

## **Key Insight**:
- The **Refined and Refined + PCA models** outperform the **Original model** in accuracy, precision, and F1 score, while still maintaining **strong recall (0.88)**.
- The **Original model** achieved the highest recall (**0.91**), but at the cost of **lower precision (0.73)** and more **false positives** (33).
- The **Refined model** offered a **better balance**, improving precision while maintaining high recall.
- **PCA did not offer any significant performance gains** compared to the **Refined model**, confirming that feature selection alone is sufficient.

## Visual Outputs

- **ROC AUC Score** ≈ 0.849 for refined models, indicating strong model discriminative power.
- **PR Curves** show a **good balance** between **sensitivity** (recall) and **specificity** (precision).
- **Coefficient analysis** reveals the impact of key features like `Glucose` and `BMI` on predicting diabetes risk.

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

## Author
Developed by Amanjot Kaur as a demonstration of ML in healthcare decision support systems.
