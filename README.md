# Diabetes Risk Prediction — ML Classification with Threshold Tuning

> End-to-end ML pipeline for diabetes screening with class imbalance handling, threshold optimization, and model interpretability.

---

## Problem Statement

Predict whether a patient is diabetic (`1`) or non-diabetic (`0`) using clinical and demographic indicators.

Core challenge: **severe class imbalance** (~6.5% diabetic). Accuracy alone is misleading — this project prioritizes recall-oriented evaluation suited for medical screening.

---

## Project Structure
```
diabetes-risk-prediction/
│
├── diabetes_prediction.py
├── diabd.csv
├── requirements.txt
└── outputs/
    ├── class_distribution.png
    ├── precision_recall.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Target | `diabetic` (0 = No, 1 = Yes) |
| Class split | ~93.5% Non-diabetic / ~6.5% Diabetic |
| Total records | 5,300+ patient records |
| Source | [add dataset link here] |

**Features:** glucose, BMI, weight, age, blood pressure (systolic/diastolic), pulse rate, hypertension, cardiovascular history, family history, gender

---

## Workflow
```
Load & Validate → EDA → Train/Test Split → Feature Scaling
    → Logistic Regression (GridSearchCV) → Threshold Tuning
    → Feature Interpretability → Random Forest (Benchmark)
```

---

## Key Design Decisions

### Class Imbalance
The dataset has a 14:1 ratio of non-diabetic to diabetic cases. Using raw accuracy here would be meaningless — a model that predicts everyone as non-diabetic would score 93.5% accuracy while being completely useless clinically.

Two strategies applied:
- `class_weight='balanced'` on both models forces the algorithm to penalize missed diabetic cases more heavily
- ROC-AUC used as the primary metric instead of accuracy — it measures the model's ability to rank diabetic cases above non-diabetic ones regardless of threshold

### Threshold Optimization
The default 0.5 probability threshold is not optimal for imbalanced medical data. Lower thresholds increase recall at the cost of precision — which is the right tradeoff for screening.

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.20 | 0.10 | 0.94 |
| 0.30 | 0.14 | 0.84 |
| 0.40 | 0.18 | 0.72 |
| 0.50 | 0.23 | 0.65 |

**Final threshold: 0.30** — in medical screening, missing a diabetic patient carries far higher cost than a false positive that gets ruled out on follow-up.

---

## Results

### Logistic Regression (Primary — Threshold = 0.30)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.83 |
| Recall (Diabetic) | 84% |
| Precision (Diabetic) | 14% |

### Random Forest (Benchmark)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.85 |
| Recall (Diabetic) | 60% |
| Precision (Diabetic) | 25% |

---

## Model Analysis

**Why Logistic Regression over Random Forest?**

Random Forest achieves a marginally higher ROC-AUC (0.85 vs 0.83) but its recall on the diabetic class drops to 60% at comparable thresholds — meaning it misses 40% of diabetic patients. Logistic Regression at threshold 0.30 catches 84% of diabetic cases, which is the clinically relevant metric for a screening tool.

The precision of 14% means roughly 1 in 7 flagged patients is actually diabetic. In a screening context this is acceptable — flagged patients would undergo further testing rather than immediate treatment.

**Feature Importance Findings**

The top risk factors identified by logistic regression coefficients:

1. **Hypertension** — strongest single predictor, aligns with known comorbidity between hypertension and diabetes
2. **Glucose level** — expected, directly related to insulin resistance
3. **Diastolic blood pressure** — vascular health indicator
4. **Weight and BMI** — obesity is a primary risk factor
5. **Age** — risk increases with age

These results align precisely with established clinical literature, which increases confidence that the model is learning genuine medical signal rather than noise.

**On Precision Being Low**

14% precision looks poor in isolation but is expected and appropriate here. With only 6.5% positive cases in the population, even a well-calibrated model at a low threshold will generate many false positives. The goal is to not miss sick patients, not to be conservative.

---

## Visualizations

**Class Distribution**
![class_distribution](outputs/class_distribution.png)

**Precision–Recall Tradeoff**
![precision_recall](outputs/precision_recall.png)

**ROC Curve**
![roc_curve](outputs/roc_curve.png)

**Top Risk Factors**
![feature_importance](outputs/feature_importance.png)

---

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## How to Run
```bash
pip install -r requirements.txt
python diabetes_prediction.py
```

---

## requirements.txt
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
