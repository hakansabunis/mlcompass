import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"
TARGET_COL = "Class"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(TRAIN_PATH)

    # Basic validation
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Ensure numeric types
    X = X.apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(y, errors="raise")

    # Candidate models
    candidates = {
        "svc_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(C=5.0, kernel="rbf", gamma="scale", probability=False, random_state=42)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Select best model with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = model

    # Fit best model on full data
    best_model.fit(X, y)

    # Save model and metadata
    artifact = {
        "model": best_model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "best_cv_accuracy": best_score,
        "best_model_name": best_name,
    }
    joblib.dump(artifact, MODEL_PATH)

    # Optional: print a small training summary
    train_pred = best_model.predict(X)
    train_acc = accuracy_score(y, train_pred)
    print(json.dumps(
        {
            "saved_model": os.path.abspath(MODEL_PATH),
            "best_model": best_name,
            "cv_accuracy": best_score,
            "train_accuracy": float(train_acc),
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
        },
        indent=2
    ))


if __name__ == "__main__":
    main()
