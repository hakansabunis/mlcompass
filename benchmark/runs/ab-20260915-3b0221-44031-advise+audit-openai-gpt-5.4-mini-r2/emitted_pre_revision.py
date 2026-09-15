import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
META_PATH = "trained_model_meta.json"
RANDOM_STATE = 42


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()

    # Basic numeric preprocessing
    # A strong tree-based model, with light target transformation for stability.
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
            ("regressor", TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )),
        ]
    )

    # Optional quick CV estimate for sanity
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    neg_rmse_scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    rmse_scores = -neg_rmse_scores
    cv_rmse_mean = float(rmse_scores.mean())
    cv_rmse_std = float(rmse_scores.std())

    # Fit on full data
    model.fit(X, y)

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_columns": feature_cols,
        "target_column": target_col,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "model_file": MODEL_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(META_PATH)}")
    print(f"5-fold CV RMSE: {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}")


if __name__ == "__main__":
    main()
