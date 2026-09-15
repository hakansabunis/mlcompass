import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Basic cleanup: ensure numeric types
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with any missing values if present
    data = pd.concat([X, y.rename(target_col)], axis=1).dropna()
    X = data[feature_cols]
    y = data[target_col].astype(int)

    # Handle possible label encodings robustly
    classes = sorted(y.unique())
    if len(classes) != 2:
        raise ValueError(f"Expected binary classification, found classes: {classes}")

    # Model choice: robust, works well on small tabular data with duplicates/outliers
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
    )

    # Quick cross-validated estimate for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        oof_pred = (oof_proba >= 0.5).astype(int)
        acc = accuracy_score(y, oof_pred)
        auc = roc_auc_score(y, oof_proba)
        print(f"CV Accuracy: {acc:.4f}")
        print(f"CV ROC AUC:  {auc:.4f}")
    except Exception as e:
        print(f"Cross-validation skipped due to: {e}")

    # Fit final model on all data
    model.fit(X, y)

    # Package everything needed for later prediction
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "classes": classes,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
