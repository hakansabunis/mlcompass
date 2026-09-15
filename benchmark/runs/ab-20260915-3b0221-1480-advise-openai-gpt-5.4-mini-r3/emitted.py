import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"
    model_path = os.path.join(os.getcwd(), "trained_model.joblib")

    df = pd.read_csv(train_path)

    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Model: robust baseline for small tabular binary classification
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # Fit final model on all training data
    model.fit(X, y)

    # Optional quick internal validation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred_cv = (proba_cv >= 0.5).astype(int)

    acc = accuracy_score(y, pred_cv)
    try:
        auc = roc_auc_score(y, proba_cv)
    except Exception:
        auc = float("nan")

    print(f"CV accuracy: {acc:.4f}")
    print(f"CV ROC AUC:   {auc:.4f}")

    # Save fitted model
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
