import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed2-r3\train.csv"
TARGET_COL = "price"
MODEL_PATH = "trained_model.joblib"


def load_data(path: str):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(float)
    return X, y, df.columns.tolist()


def build_model(random_state: int = 42):
    # A strong default for this tabular regression problem
    xgb = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.5,
        gamma=0.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb),
        ]
    )
    return pipe


def main():
    X, y, feature_cols = load_data(DATA_PATH)

    # Small validation split for a quick sanity check
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = build_model(random_state=42)

    # Cross-validation gives a more stable estimate than a single split
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    print(f"CV RMSE: mean={cv_rmse.mean():.6f}, std={cv_rmse.std():.6f}")

    # Fit on train split, evaluate on validation split
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"Holdout RMSE: {val_rmse:.6f}")

    # Refit on full dataset for final model
    final_model = build_model(random_state=42)
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
