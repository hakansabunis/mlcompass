import os
import json
import random

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
META_PATH = "trained_model_meta.json"
RANDOM_STATE = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(RANDOM_STATE)

    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()

    # Hold-out split for a quick validation estimate
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                ),
            ),
        ]
    )

    # Fit on training split and evaluate on validation split
    model.fit(X_train, y_train)
    valid_preds = model.predict(X_valid)
    valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)

    # Refit on full data before saving
    model.fit(X, y)

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_columns": feature_cols,
        "target_column": target_col,
        "random_state": RANDOM_STATE,
        "validation_rmse": float(valid_rmse),
        "model_file": MODEL_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(META_PATH)}")
    print(f"Hold-out RMSE: {valid_rmse:.6f}")


if __name__ == "__main__":
    main()
