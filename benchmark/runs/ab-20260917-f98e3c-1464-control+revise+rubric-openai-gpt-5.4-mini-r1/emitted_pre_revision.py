import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def load_data(path: str):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in training data.")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_model(random_state: int = 42):
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(random_state=random_state)

    lr = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class="auto",
        random_state=random_state,
    )

    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
            ("lr", lr),
        ],
        voting="soft",
        n_jobs=-1,
        weights=[3, 3, 2, 1],
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return pipeline


def main():
    X, y = load_data(TRAIN_PATH)

    model = build_model(random_state=42)

    # Optional quick validation for sanity
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score), n_jobs=-1)
        print(f"CV accuracy: {scores.mean():.6f} +/- {scores.std():.6f}")
    except Exception as e:
        print(f"Cross-validation skipped due to error: {e}")

    # Fit final model on full data
    model.fit(X, y)

    # Save fitted model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")

    # Save basic metadata for reference
    metadata = {
        "feature_columns": list(X.columns),
        "target_column": "Class",
        "model_file": MODEL_PATH,
    }
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to: model_metadata.json")


if __name__ == "__main__":
    main()
