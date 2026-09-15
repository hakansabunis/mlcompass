import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
TARGET_COL = "Class"
MODEL_PATH = "trained_model.joblib"


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleanup
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_cols),
            ("cat", categorical_preprocess, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=300,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Quick internal evaluation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: mean={acc_scores.mean():.4f}, std={acc_scores.std():.4f}")

    # ROC AUC if possible
    try:
        auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        print(f"CV ROC AUC:  mean={auc_scores.mean():.4f}, std={auc_scores.std():.4f}")
    except Exception:
        pass

    # Fit final model on all data
    pipeline.fit(X, y)

    # Save model and metadata
    artifact = {
        "pipeline": pipeline,
        "target_col": TARGET_COL,
        "feature_columns": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")

    # Optional: save a small metadata JSON for convenience
    meta_path = os.path.splitext(MODEL_PATH)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "feature_columns": X.columns.tolist(),
                "categorical_cols": categorical_cols,
                "numeric_cols": numeric_cols,
                "cv_accuracy_mean": float(acc_scores.mean()),
                "cv_accuracy_std": float(acc_scores.std()),
            },
            f,
            indent=2,
        )
    print(f"Saved metadata to: {os.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()
