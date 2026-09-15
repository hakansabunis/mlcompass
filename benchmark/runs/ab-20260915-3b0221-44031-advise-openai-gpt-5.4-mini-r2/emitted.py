import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
TARGET = "price"
MODEL_OUT = "trained_model.joblib"
METRICS_OUT = "training_metrics.json"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")
    return df


def build_model() -> TransformedTargetRegressor:
    feature_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=600,
                    max_depth=6,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    max_bins=255,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Target is positive and appears skewed/capped; log1p can improve fit stability.
    model = TransformedTargetRegressor(
        regressor=feature_model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return model


def main():
    df = load_data(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    mae = mean_absolute_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)

    # Refit on full data for final model
    final_model = build_model()
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_OUT)

    metrics = {
        "data_path": DATA_PATH,
        "target": TARGET,
        "model_file": os.path.abspath(MODEL_OUT),
        "validation": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid)),
        },
        "features": list(X.columns),
        "random_state": RANDOM_STATE,
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_OUT)}")
    print(f"Saved metrics to: {os.path.abspath(METRICS_OUT)}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MAE : {mae:.6f}")
    print(f"Validation R2   : {r2:.6f}")


if __name__ == "__main__":
    main()
