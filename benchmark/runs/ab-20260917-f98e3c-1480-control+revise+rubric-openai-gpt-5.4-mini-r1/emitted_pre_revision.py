import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

warnings.filterwarnings("ignore")

TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"

RANDOM_STATE = 42


def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    df = pd.read_csv(TRAIN_PATH)

    target_col = "Class"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Ensure categorical columns are strings / missing handled consistently
    for c in categorical_cols:
        X[c] = X[c].astype("string")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot()),
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

    # Base model
    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        learning_rate=0.05,
        max_iter=300,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Small hyperparameter search for robustness
    param_distributions = {
        "model__learning_rate": np.linspace(0.02, 0.15, 8),
        "model__max_iter": [150, 200, 250, 300, 400],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [5, 10, 20, 30, 50],
        "model__l2_regularization": np.logspace(-6, 0, 7),
    }

    # Use a quick stratified split for validation if possible
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    # Refit on full data using best found params
    final_model = search.best_estimator_
    final_model.fit(X, y)

    # Save model and metadata
    bundle = {
        "model": final_model,
        "columns": list(X.columns),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_col": target_col,
        "validation_accuracy": float(val_acc),
        "best_params": search.best_params_,
    }
    joblib.dump(bundle, MODEL_PATH)

    # Also save a small text metadata file for convenience
    meta = {
        "validation_accuracy": float(val_acc),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "columns": list(X.columns),
    }
    with open("trained_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Validation accuracy: {val_acc:.6f}")
    print(f"Best params: {search.best_params_}")


if __name__ == "__main__":
    main()
