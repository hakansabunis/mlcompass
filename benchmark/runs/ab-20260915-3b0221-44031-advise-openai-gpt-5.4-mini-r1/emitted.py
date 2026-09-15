import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col].astype(float)

    # Hold-out split for a quick sanity check
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # A strong default for tabular regression with numeric features
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=None,
                    max_iter=500,
                    min_samples_leaf=20,
                    l2_regularization=0.0,
                    random_state=42,
                ),
            ),
        ]
    )

    # Optional cross-validation on training split
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    # Fit on training split and evaluate on validation split
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)

    rmse = mean_squared_error(y_valid, valid_pred, squared=False)
    mae = mean_absolute_error(y_valid, valid_pred)

    # Refit on full dataset before saving
    model.fit(X, y)

    # Save model plus metadata needed for later use
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "holdout_rmse": float(rmse),
        "holdout_mae": float(mae),
    }
    joblib.dump(artifact, MODEL_PATH)

    # Print a small training summary
    print(json.dumps(
        {
            "saved_model": os.path.abspath(MODEL_PATH),
            "cv_rmse_mean": artifact["cv_rmse_mean"],
            "cv_rmse_std": artifact["cv_rmse_std"],
            "holdout_rmse": artifact["holdout_rmse"],
            "holdout_mae": artifact["holdout_mae"],
            "n_rows": int(df.shape[0]),
            "n_features": int(len(feature_cols)),
        },
        indent=2
    ))


if __name__ == "__main__":
    main()
