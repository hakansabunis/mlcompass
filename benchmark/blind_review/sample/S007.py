"""Train a classifier on the provided CSV and save the fitted model.

Run:  python train_model.py
Output: model.joblib in the current working directory.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

CSV_PATH = Path(
    r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"
)
TARGET = "Class"
MODEL_OUT = Path.cwd() / "model.joblib"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2
SCORING = "balanced_accuracy"

# --- Global reproducibility ---------------------------------------------------
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_candidates() -> dict:
    """Return {name: (estimator, parameter_grid)}."""
    return {
        "logreg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "svc": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(probability=True, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "clf__C": [0.5, 1.0, 5.0, 20.0],
                "clf__gamma": ["scale", 0.05, 0.1, 0.5],
            },
        ),
        "knn": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "clf__n_neighbors": [3, 5, 7, 11, 15],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300, 600],
                "max_depth": [None, 4, 6, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "et": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300, 600],
                "max_depth": [None, 4, 6, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "gb": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} not found in {df.columns.tolist()}")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    features = [c for c in df.columns if c != TARGET]
    if not features:
        raise ValueError("No feature columns available.")

    X_all = df[features].astype(float)
    y_all = df[TARGET].astype(int)

    # Duplicate rows (~24% of the data) would leak between CV folds and inflate
    # the validation scores, so model selection is done on de-duplicated data.
    dedup = df.drop_duplicates().reset_index(drop=True)
    X_dedup = dedup[features].astype(float)
    y_dedup = dedup[TARGET].astype(int)

    print(f"Rows: {len(df)} (unique: {len(dedup)}) | Features: {features}")
    print(f"Class counts: {y_all.value_counts().to_dict()}")

    # Explicit held-out validation split for honest model comparison.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dedup,
        y_dedup,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_dedup,
    )
    print(
        f"Held-out validation split: train={len(X_tr)} val={len(X_val)} "
        f"(test_size={TEST_SIZE})"
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best_name: str | None = None
    best_cv_score: float = -np.inf
    best_val_score: float = -np.inf
    best_estimator = None

    for name, (estimator, grid) in build_candidates().items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_tr, y_tr)
        cv_score = float(search.best_score_)
        val_score = float(search.best_estimator_.score(X_val, y_val))
        print(
            f"  {name:<7} cv_{SCORING}={cv_score:.4f} "
            f"val_{SCORING}={val_score:.4f}  best_params={search.best_params_}"
        )
        if val_score > best_val_score or (
            val_score == best_val_score and cv_score > best_cv_score
        ):
            best_val_score = val_score
            best_cv_score = cv_score
            best_name = name
            best_estimator = search.best_estimator_

    if best_estimator is None:
        raise RuntimeError("Model selection failed: no candidate could be trained.")

    print(
        f"\nSelected model: {best_name} "
        f"(cv={best_cv_score:.4f}, held-out val={best_val_score:.4f})"
    )

    # Refit the winning configuration on all available rows (duplicates included).
    best_estimator.fit(X_all, y_all)

    joblib.dump(best_estimator, MODEL_OUT)
    print(f"Saved fitted model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
