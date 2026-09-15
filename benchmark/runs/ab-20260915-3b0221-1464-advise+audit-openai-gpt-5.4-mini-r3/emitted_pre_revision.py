import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"
    df = pd.read_csv(train_path)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Basic cleaning: ensure numeric types
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(y, errors="coerce").astype(int)

    # Model pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                    min_samples_leaf=1,
                    min_samples_split=2,
                ),
            ),
        ]
    )

    # Optional validation estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit final model on all data
    model.fit(X, y)

    # Save model and metadata in current working directory
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    out_path = os.path.join(os.getcwd(), "trained_model.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
