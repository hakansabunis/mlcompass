import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
MODEL_PATH = "price_model.joblib"
TARGET_COLUMN = "price"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    valid_preds = pipeline.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
    r2 = r2_score(y_valid, valid_preds)

    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation R2: {r2:.6f}")

    # Refit on the full dataset before saving the final model.
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved fitted model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
