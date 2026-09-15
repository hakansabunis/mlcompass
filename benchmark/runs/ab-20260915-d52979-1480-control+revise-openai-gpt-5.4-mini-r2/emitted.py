import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
MODEL_PATH = "trained_model.joblib"


def make_one_hot_encoder():
    # Compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(X: pd.DataFrame) -> VotingClassifier:
    preprocessor = build_preprocessor(X)

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
    )

    et = ExtraTreesClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=1,
    )

    estimators = [
        ("rf", Pipeline([("prep", preprocessor), ("model", rf)])),
        ("et", Pipeline([("prep", preprocessor), ("model", et)])),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1, 1],
        n_jobs=-1,
    )


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in the dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Sanity-check validation score before fitting on full data
    stratify = y if y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, valid_pred)
    f1 = f1_score(y_valid, valid_pred, average="weighted")

    print(f"Validation accuracy: {acc:.6f}")
    print(f"Validation weighted F1: {f1:.6f}")

    # Refit final model on all data
    final_model = build_model(X)
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "feature_columns": X.columns.tolist(),
        "target_column": "Class",
        "validation_accuracy": float(acc),
        "validation_weighted_f1": float(f1),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
