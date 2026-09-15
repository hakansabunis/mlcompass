import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(RANDOM_SEED)

    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    # Remove exact duplicates to reduce leakage from repeated rows
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Hold out a validation split for an honest evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Build a robust classification pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                    min_samples_leaf=1,
                    min_samples_split=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Validation performance
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Optional cross-validation sanity check on the full de-duplicated dataset
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    n_splits = max(2, min(5, min_class_count))
    if n_splits >= 2 and len(y) >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        # Reuse model on full data for final fit after reporting CV diagnostics
        cv_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            fold_model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=RANDOM_SEED,
                            class_weight="balanced",
                            min_samples_leaf=1,
                            min_samples_split=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            fold_model.fit(X_tr, y_tr)
            cv_scores.append(accuracy_score(y_te, fold_model.predict(X_te)))
        cv_scores = np.array(cv_scores, dtype=float)
        print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit final model on all available data
    model.fit(X, y)

    # Save model and metadata for later inference
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
