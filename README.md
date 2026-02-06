# Diabetes Risk Prediction using Machine Learning

## Project Overview
This project builds an end-to-end diabetes risk prediction system using clinical and demographic data.  
The emphasis is not only on model performance, but also on handling real-world challenges such as class imbalance, decision-threshold tuning, and model interpretability.

Two models are developed and compared:
- Logistic Regression (optimized and interpretable)
- Random Forest (non-linear benchmark)

---

## Problem Statement
Predict whether a patient is diabetic (1) or non-diabetic (0) using medical and lifestyle indicators.

A key challenge in this dataset is severe class imbalance, with diabetic cases forming a small minority.  
Therefore, accuracy alone is not a reliable metric, and recall-oriented evaluation is required.

---

## Dataset Summary
- Target variable: `diabetic` (0 = No, 1 = Yes)
- Class distribution:
  - Non-diabetic: ~93.5%
  - Diabetic: ~6.5%

### Features include:
- Glucose level
- Blood pressure (systolic and diastolic)
- BMI, weight, age
- Hypertension and cardiovascular conditions
- Family medical history

---

## Exploratory Data Analysis (EDA)
A quick EDA step highlights the heavy class imbalance in the target variable.  
This motivates:
- Use of `class_weight='balanced'`
- ROC-AUC and Precision–Recall based evaluation instead of accuracy

---

## Modeling Approach

### Logistic Regression (Primary Model)
Logistic Regression was chosen as the primary model due to:
- Interpretability
- Stability on tabular medical data
- Ease of explaining risk factors to stakeholders

Techniques applied:
- Feature scaling using StandardScaler
- Class imbalance handling with `class_weight='balanced'`
- Hyperparameter tuning using GridSearchCV
- Evaluation using ROC-AUC
- Decision threshold optimization

---

## Threshold Optimization
Instead of using the default probability threshold of 0.5, multiple thresholds were evaluated to balance precision and recall.

| Threshold | Precision | Recall |
|----------|----------|--------|
| 0.20 | 0.10 | 0.94 |
| 0.30 | 0.14 | 0.84 |
| 0.40 | 0.18 | 0.72 |
| 0.50 | 0.23 | 0.65 |

Final threshold selected: **0.30**

### Rationale
In medical screening use cases, missing a diabetic patient is more costly than generating a false positive.  
Therefore, recall was prioritized over precision.

---

## Final Logistic Regression Performance (Threshold = 0.30)

- ROC-AUC: **0.83**
- Recall (Diabetic class): **84%**
- Precision (Diabetic class): **14%**

This reflects a realistic trade-off expected in healthcare screening models.

---

## Model Interpretability
Logistic Regression coefficients were analyzed to understand feature impact.

Top risk factors identified:
1. Hypertension
2. Glucose level
3. Diastolic blood pressure
4. Weight
5. BMI
6. Age

These results align well with established clinical knowledge, increasing trust in the model.

---

## Random Forest (Benchmark Model)
A Random Forest classifier was trained as a non-linear benchmark.

### Performance:
- ROC-AUC: **0.85**
- Recall (Diabetic class): **60%**
- Precision (Diabetic class): **25%**

### Observations:
- Slightly higher ROC-AUC than Logistic Regression
- Lower recall at comparable thresholds
- Reduced interpretability

Logistic Regression was preferred for its balance between performance and explainability.

---

## Visualizations Included
- Class distribution plot (imbalance analysis)
- Precision–Recall vs Threshold curve
- Logistic Regression feature importance
- Random Forest feature importance

Only high-impact visualizations were included to maintain clarity.

---

## Conclusion
This project demonstrates a realistic machine learning workflow with emphasis on:
- Handling class imbalance
- Appropriate metric selection
- Threshold tuning based on business context
- Model interpretability

The approach mirrors real-world healthcare analytics rather than optimizing for superficial accuracy metrics.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## How to Run
```bash
pip install -r requirements.txt
python diabetes_prediction.py

