import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv"
MODEL_PATH = os.path.join(os.getcwd(), "trained_model.joblib")
META_PATH = os.path.join(os.getcwd(), "trained_model_meta.json")
RANDOM_STATE = 42


def build_model():
    categorical_features = ["V2"]
    numeric_features = ["V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )
    return model


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleanup
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Ensure binary labels are consistent and integer-like
    y = pd.Series(y).astype(int)

    model = build_model()

    # Small dataset: use cross-validation for a sanity check
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    try:
        auc_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=None,
        )
        acc_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )
        metrics = {
            "cv_roc_auc_mean": float(np.mean(auc_scores)),
            "cv_roc_auc_std": float(np.std(auc_scores)),
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "n_rows": int(len(df)),
            "n_duplicates_dropped": int(len(pd.read_csv(DATA_PATH)) - len(df)),
        }
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Cross-validation skipped due to error: {e}")

    # Fit final model on all data
    model.fit(X, y)

    # Training-set check for a quick sanity metric
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        try:
            train_auc = roc_auc_score((y == sorted(y.unique())[1]).astype(int), proba)
            print(f"Training ROC AUC: {train_auc:.4f}")
        except Exception:
            pass
    preds = model.predict(X)
    try:
        train_acc = accuracy_score(y, preds)
        print(f"Training accuracy: {train_acc:.4f}")
    except Exception:
        pass

    # Save fitted model
    joblib.dump(model, MODEL_PATH)

    # Save metadata for future reference
    meta = {
        "model_path": MODEL_PATH,
        "target": "Class",
        "feature_columns": list(X.columns),
        "dropped_duplicates": int(len(pd.read_csv(DATA_PATH)) - len(df)),
        "random_state": RANDOM_STATE,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()
