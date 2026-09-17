import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"
TARGET_COL = "price"
MODEL_FILENAME = "price_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found in {DATA_PATH}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(float)

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=20260915
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=1000,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    random_state=20260915,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R2: {r2:.4f}")

    pipeline.fit(X, y)

    model_path = Path.cwd() / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    print(f"Saved fitted model to {model_path}")

if __name__ == "__main__":
    main()
