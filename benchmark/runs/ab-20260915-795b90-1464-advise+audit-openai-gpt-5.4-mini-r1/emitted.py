import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"
SEED = 42


def main():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Ensure numeric types
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with missing values if any
    data = pd.concat([X, y.rename(target_col)], axis=1).dropna()
    X = data[feature_cols]
    y = data[target_col].astype(int)

    classes = sorted(y.unique())
    if len(classes) != 2:
        raise ValueError(f"Expected binary classification, found classes: {classes}")

    # Hold-out validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=SEED,
        class_weight="balanced",
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
    )

    # Fit on training split and evaluate on validation split
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_proba)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC AUC:  {auc:.4f}")

    # Refit final model on all available data
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "classes": classes,
        "seed": SEED,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
