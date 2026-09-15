import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC

# Path to the training data
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"
model_path = os.path.join(os.getcwd(), "model.joblib")

# Load data
df = pd.read_csv(csv_path)
X = df.drop(columns=["Class"])
y = df["Class"].values

# Preprocessing pipeline
preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Candidate models
candidates = {
    "logreg": LogisticRegression(max_iter=5000, random_state=42),
    "logreg_balanced": LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=42
    ),
    "rf": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "rf_balanced": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "gb": GradientBoostingClassifier(random_state=42),
    "hgb": HistGradientBoostingClassifier(random_state=42),
    "svc": SVC(probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = -np.inf
best_pipeline = None
best_name = None

for name, clf in candidates.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_pipeline = pipeline
        best_name = name

print(f"Best model: {best_name} (CV accuracy: {best_score:.4f})")

# Fit the best model on the full training data
best_pipeline.fit(X, y)

# Save the fitted model
joblib.dump(best_pipeline, model_path)
print(f"Model saved to {model_path}")
