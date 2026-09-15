import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
TARGET_COL = "Class"

NUMERIC_COLS = ["V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]
CATEGORICAL_COLS = ["V2"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found in {DATA_PATH}")

    # Drop rows with missing target labels (cannot train on those).
    df = df.dropna(subset=[TARGET_COL])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
