import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
METADATA_PATH = "trained_model_metadata.json"
TARGET_COL = "Class"

warnings.filterwarnings("ignore")


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = None
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores)), "roc_auc"
    except Exception:
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores)), "accuracy"


def main():
    df = load_data(TRAIN_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataset.")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # Ensure target is integer-coded if possible
    y = pd.Series(y).astype(int)

    preprocessor = build_preprocessor(X)

    candidate_models = []

    # Strong baseline for tabular data
    candidate_models.append(
        (
            "logreg",
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        )
    )

    candidate_models.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
            ),
        )
    )

    candidate_models.append(
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=800,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                min_samples_leaf=1,
            ),
        )
    )

    candidate_models.append(
        (
            "gb",
            GradientBoostingClassifier(random_state=42),
        )
    )

    best_name = None
    best_score = -np.inf
    best_scoring = None
    best_pipe = None

    for name, clf in candidate_models:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )

        score, scoring_name = evaluate_model(pipe, X, y)
        print(f"{name}: {scoring_name} = {score:.6f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_scoring = scoring_name
            best_pipe = pipe

    # Fit the best model on full data
    best_pipe.fit(X, y)

    # Optional calibration for classifiers with predict_proba if not logistic regression
    # Keep final output simple and robust: trained pipeline is saved directly.
    joblib.dump(best_pipe, MODEL_PATH)

    metadata = {
        "model_path": MODEL_PATH,
        "target": TARGET_COL,
        "best_model": best_name,
        "best_cv_score": best_score,
        "scoring": best_scoring,
        "features": list(X.columns),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(METADATA_PATH)}")


if __name__ == "__main__":
    main()
