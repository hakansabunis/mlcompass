import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# Paths
TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    # Load data
    df = pd.read_csv(TRAIN_PATH)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Preprocessing
    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Main model
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=600,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Optional sanity-check CV
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=rmse_scorer, n_jobs=-1)
    cv_rmse = float((-cv_scores).mean())
    cv_rmse_std = float(cv_scores.std())

    # Fit on full data
    pipeline.fit(X, y)

    # Save model and metadata
    payload = {
        "model": pipeline,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cv_rmse": cv_rmse,
        "cv_rmse_std": cv_rmse_std,
    }
    joblib.dump(payload, MODEL_PATH)

    # Also save a small metadata JSON for convenience
    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cv_rmse": cv_rmse,
        "cv_rmse_std": cv_rmse_std,
        "model_path": MODEL_PATH,
    }
    with open("trained_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"5-fold CV RMSE: {cv_rmse:.6f} (+/- {cv_rmse_std:.6f})")

if __name__ == "__main__":
    main()
