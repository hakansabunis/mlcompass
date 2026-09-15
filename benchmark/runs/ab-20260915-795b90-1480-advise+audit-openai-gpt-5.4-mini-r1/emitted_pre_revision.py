import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"
META_PATH = "trained_model_metadata.json"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Basic cleanup / type handling
    X["V2"] = X["V2"].astype(str)

    numeric_features = [c for c in X.columns if c != "V2"]
    categorical_features = ["V2"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Cross-validation for a quick sanity check
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    try:
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        cv_summary = {
            "accuracy_mean": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
            "roc_auc_mean": float(np.mean(auc_scores)),
            "roc_auc_std": float(np.std(auc_scores)),
        }
    except Exception:
        # If ROC AUC fails for any reason, still proceed with training.
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        cv_summary = {
            "accuracy_mean": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
        }

    # Fit final model on all available data
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, MODEL_PATH)

    # Save minimal metadata
    metadata = {
        "data_path": DATA_PATH,
        "target_col": target_col,
        "features": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cv_summary": cv_summary,
        "classes_": [int(c) if hasattr(c, "__int__") else c for c in clf.named_steps["model"].classes_],
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(META_PATH)}")
    print("Cross-validation summary:")
    for k, v in cv_summary.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
