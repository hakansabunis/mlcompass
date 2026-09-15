import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
    model_path = "trained_model.joblib"

    # Load data
    df = pd.read_csv(train_path)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Ensure numeric types
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce").astype(int)

    # Class labels may be 1/2; LogisticRegression handles this directly.
    # Use a simple, robust pipeline for small tabular data.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=2000,
                ),
            ),
        ]
    )

    # Optional validation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

    print("Cross-validation accuracy: {:.4f} ± {:.4f}".format(acc_scores.mean(), acc_scores.std()))
    print("Cross-validation f1_macro: {:.4f} ± {:.4f}".format(f1_scores.mean(), f1_scores.std()))

    # Fit final model on all data
    model.fit(X, y)

    # Save fitted model and metadata
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "class_labels": sorted(y.unique().tolist()),
    }
    joblib.dump(artifact, model_path)

    print(f"Saved model to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()
