import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
TARGET_COL = "Class"
MODEL_PATH = "trained_model.joblib"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    return df


def main():
    df = load_data(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # A robust, simple baseline that handles duplicates and requires no feature scaling.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    # Optional quick cross-validation to fit a bit more confidence into the script.
    # This does not affect final training; it is informational only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Fit final model on all data
    model.fit(X, y)

    # Save model artifact
    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "target_col": TARGET_COL,
        "classes_": sorted(y.unique().tolist()),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
