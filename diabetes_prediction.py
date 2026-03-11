# =============================
# 1. IMPORT LIBRARIES
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support

# =============================
# 2. LOAD & PREPROCESS DATA
# =============================
df = pd.read_csv(r"C:\Users\babul\OneDrive\Desktop\diabd.csv")

print("Missing values per column:")
print(df.isnull().sum())
print()

df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

X = df.drop('diabetic', axis=1)
y = df['diabetic']

# =============================
# 3. EDA — CLASS DISTRIBUTION
# =============================
print(y.value_counts(normalize=True))

plt.figure(figsize=(4, 4))
sns.countplot(x=y)
plt.title("Class Distribution (Diabetes)")
plt.xlabel("Diabetic (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =============================
# 4. TRAIN–TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================
# 5. FEATURE SCALING
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_rf = X_train_scaled
X_test_rf  = X_test_scaled

# =============================
# 6. LOGISTIC REGRESSION
# =============================
param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)
lr_model = grid.best_estimator_

print("\nBest Logistic Regression Params:", grid.best_params_)

# =============================
# 7. THRESHOLD OPTIMIZATION
# =============================
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nThreshold | Precision | Recall")
print("-------------------------------")
for t in [0.2, 0.3, 0.4, 0.5]:
    y_pred_t = (y_prob >= t).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_test, y_pred_t, average=None)
    print(f"{t:9} | {p[1]:9.2f} | {r[1]:6.2f}")

FINAL_THRESHOLD = 0.3
y_pred_final = (y_prob >= FINAL_THRESHOLD).astype(int)

print(f"\nFinal Logistic Model @ Threshold {FINAL_THRESHOLD}")
print(classification_report(y_test, y_pred_final))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =============================
# 8. PRECISION-RECALL TRADEOFF
# =============================
thresholds = np.linspace(0.1, 0.9, 20)
precisions, recalls = [], []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_test, y_pred_t, average='binary')
    precisions.append(p)
    recalls.append(r)

plt.figure(figsize=(5, 4))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.axvline(FINAL_THRESHOLD, linestyle="--", label=f"Chosen Threshold = {FINAL_THRESHOLD}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Tradeoff")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 9. ROC CURVE
# =============================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 10. FEATURE IMPORTANCE (LR)
# =============================
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values(by='coefficient', key=abs, ascending=False)

print("\nLogistic Regression Feature Importance")
print(coef_df)

plt.figure(figsize=(6, 4))
sns.barplot(x='coefficient', y='feature', data=coef_df.head(8))
plt.title("Top Risk Factors for Diabetes (Logistic Regression)")
plt.tight_layout()
plt.show()

# =============================
# 11. RANDOM FOREST (BENCHMARK)
# =============================
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

rf_param_grid = {
    'n_estimators': [200],
    'max_depth': [5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

rf_grid = GridSearchCV(
    rf, rf_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train_rf, y_train)
rf_best = rf_grid.best_estimator_

rf_prob = rf_best.predict_proba(X_test_rf)[:, 1]
rf_pred = rf_best.predict(X_test_rf)

print("\nRandom Forest Results")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

rf_feat = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nRandom Forest Feature Importance")
print(rf_feat)
