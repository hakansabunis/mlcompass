import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()

    # Basic train/validation split for lightweight model selection
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: impute missing values + model
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )

    # Small randomized search over a reasonably strong regressor
    param_distributions = {
        "model__learning_rate": np.linspace(0.02, 0.2, 10),
        "model__max_iter": [200, 300, 500, 800],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate on validation split
    val_pred = best_model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    r2 = r2_score(y_val, val_pred)

    # Refit best model on the full dataset
    best_model.fit(X, y)

    # Save model and metadata
    artifact = {
        "model": best_model,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "validation_rmse": float(rmse),
        "validation_r2": float(r2),
        "best_params": search.best_params_,
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation R2: {r2:.6f}")
    print(f"Best params: {json.dumps(search.best_params_, indent=2)}")


if __name__ == "__main__":
    main()
