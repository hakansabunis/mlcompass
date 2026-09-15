import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(TRAIN_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # A strong, robust baseline for tabular integer features
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=None,
    )

    # Optional quick validation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # Fit final model on all data
    model.fit(X, y)

    # Save a compact bundle with metadata
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
