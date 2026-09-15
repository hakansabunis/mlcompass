import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    # Build a robust model for a small binary classification dataset.
    # Standardization helps logistic regression with differing feature scales.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    # Optional quick validation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)

    acc = accuracy_score(y, oof_pred)
    try:
        auc = roc_auc_score(y, oof_proba)
    except Exception:
        auc = float("nan")

    print(f"Training rows: {len(df)}")
    print(f"Features: {feature_cols}")
    print(f"CV Accuracy: {acc:.4f}")
    print(f"CV ROC AUC: {auc:.4f}")

    # Fit final model on all data
    model.fit(X, y)

    # Persist model along with feature order metadata
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "target_column": target_col,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
