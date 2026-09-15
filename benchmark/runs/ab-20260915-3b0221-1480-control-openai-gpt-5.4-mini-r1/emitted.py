import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv"
TARGET_COL = "Class"
MODEL_PATH = "trained_model.joblib"
METADATA_PATH = "trained_model_metadata.json"
RANDOM_STATE = 42


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identify column types
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "string"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocessing for numeric and categorical features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
        verbose_feature_names_out=False,
    )

    # Model choice:
    # HistGradientBoostingClassifier is strong on tabular data and works well with dense inputs.
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train/validation split for a quick sanity check
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    clf.fit(X_train, y_train)

    valid_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, valid_pred)

    print(f"Validation accuracy: {acc:.6f}")
    print("Classification report:")
    print(classification_report(y_valid, valid_pred, zero_division=0))

    # Optional cross-validation on the full dataset for a more stable estimate
    if y.nunique() > 1 and len(df) >= 10:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"CV accuracy: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # Fit on full dataset before saving
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, MODEL_PATH)

    # Save basic metadata
    metadata = {
        "model_file": MODEL_PATH,
        "target_column": TARGET_COL,
        "feature_columns": list(X.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "validation_accuracy": float(acc),
        "random_state": RANDOM_STATE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Saved metadata to: {os.path.abspath(METADATA_PATH)}")


if __name__ == "__main__":
    main()
