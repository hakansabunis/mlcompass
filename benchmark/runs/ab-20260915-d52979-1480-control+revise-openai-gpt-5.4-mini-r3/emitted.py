import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def main():
    warnings.filterwarnings("ignore")

    train_path = Path(r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv")
    model_path = Path.cwd() / "trained_model.joblib"
    meta_path = Path.cwd() / "trained_model_metadata.json"

    df = pd.read_csv(train_path)

    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=None,
                max_iter=300,
                min_samples_leaf=20,
                l2_regularization=0.0,
                random_state=42,
            )),
        ]
    )

    model.fit(X, y)

    joblib.dump(model, model_path)

    meta = {
        "target": target_col,
        "features": list(X.columns),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "model_file": model_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
