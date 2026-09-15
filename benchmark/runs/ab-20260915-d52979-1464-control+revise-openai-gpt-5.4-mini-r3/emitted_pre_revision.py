import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


CSV_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    df = pd.read_csv(CSV_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # A robust, generally strong baseline for small tabular integer data
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )

    # Fit on full data
    model.fit(X, y)

    # Save model and minimal metadata
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
