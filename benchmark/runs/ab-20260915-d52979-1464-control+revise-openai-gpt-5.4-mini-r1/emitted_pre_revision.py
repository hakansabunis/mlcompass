import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Build a simple, robust classifier pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Fit on the full dataset
    model.fit(X, y)

    # Optional quick sanity check on a holdout split (does not affect saved model)
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        sanity_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        sanity_model.fit(X_train, y_train)
        preds = sanity_model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        print(f"Validation accuracy: {acc:.6f}")
    except Exception as e:
        print(f"Validation skipped: {e}")

    # Save fitted model and metadata
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
