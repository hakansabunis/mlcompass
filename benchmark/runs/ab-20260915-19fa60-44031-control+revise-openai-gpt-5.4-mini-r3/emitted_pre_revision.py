import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed2-r3\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Basic train/validation split for a light sanity check.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Strong baseline for tabular regression, robust to nonlinearity.
    base_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=500,
                    min_samples_leaf=20,
                    l2_regularization=0.0,
                    random_state=42,
                ),
            ),
        ]
    )

    # Log-transform target to help with skew and improve stability.
    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    model.fit(X_train, y_train)

    # Optional validation metric printed for visibility.
    val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"Validation RMSE: {rmse:.6f}")

    # Refit on all available data before saving.
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
