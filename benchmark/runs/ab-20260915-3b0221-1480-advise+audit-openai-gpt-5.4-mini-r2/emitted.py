import os
import json
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
META_PATH = "trained_model_metadata.json"
RANDOM_STATE = 42


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    set_seeds(RANDOM_STATE)

    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Basic cleanup: trim string columns
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str).str.strip()

    # Column groups
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocess
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=400,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Hold-out validation split for a quick reproducible sanity check
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_proba = None
    val_acc = float(accuracy_score(y_val, val_pred))

    try:
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_proba))
    except Exception:
        val_auc = None

    # Optional cross-validation on full training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    try:
        auc_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=None,
        )
        cv_auc = float(np.mean(auc_scores))
    except Exception:
        cv_auc = None

    # Refit final model on all data
    pipeline.fit(X, y)

    # Save model and metadata
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "target": target_col,
        "feature_columns": list(X.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "random_state": RANDOM_STATE,
        "holdout_accuracy": val_acc,
        "holdout_roc_auc": val_auc,
        "cv_roc_auc_mean": cv_auc,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(META_PATH)}")
    print(f"Holdout accuracy: {val_acc:.6f}")
    if val_auc is not None:
        print(f"Holdout ROC AUC: {val_auc:.6f}")
    if cv_auc is not None:
        print(f"Mean CV ROC AUC: {cv_auc:.6f}")


if __name__ == "__main__":
    main()
