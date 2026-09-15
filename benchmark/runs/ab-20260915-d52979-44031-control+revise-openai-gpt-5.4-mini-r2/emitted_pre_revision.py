import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
    model_path = "trained_model.joblib"

    # Load data
    df = pd.read_csv(train_path)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Basic cleanup: ensure numeric and handle unexpected values
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols]
    y = df[target_col]

    # Hold-out split for simple validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model pipeline
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )

    # Hyperparameter search space
    param_distributions = {
        "model__learning_rate": np.linspace(0.01, 0.2, 20),
        "model__max_depth": [None, 3, 4, 5, 6, 7, 8, 10],
        "model__max_leaf_nodes": [15, 31, 63, 127, 255],
        "model__min_samples_leaf": [5, 10, 20, 30, 50],
        "model__l2_regularization": np.logspace(-8, 1, 20),
        "model__max_bins": [64, 128, 255],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Validation performance
    val_pred = best_model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)

    # Refit best model on all data for final training
    best_model.fit(X, y)

    # Save model and metadata
    artifact = {
        "model": best_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "validation_rmse": float(rmse),
        "best_params": search.best_params_,
    }
    joblib.dump(artifact, model_path)

    print(f"Saved model to: {os.path.abspath(model_path)}")
    print(f"Validation RMSE: {rmse:.6f}")
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=2, default=str))


if __name__ == "__main__":
    main()
