import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"
METADATA_PATH = "model_metadata.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in training data.")

    # Remove duplicate rows before splitting, while preserving the target relationship.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_model(random_state: int = RANDOM_STATE):
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
    np.random.seed(RANDOM_STATE)

    X, y = load_data(TRAIN_PATH)

    # Hold out a test set for unbiased evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(random_state=RANDOM_STATE)

    # Fit only on training data.
    model.fit(X_train, y_train)

    # Evaluate on unseen data using a metric robust to class imbalance.
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Hold-out balanced accuracy: {bal_acc:.6f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=6))

    # Optional cross-validation on the training portion for additional sanity checking.
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            fold_model = build_model(random_state=RANDOM_STATE)
            fold_model.fit(X_fold_train, y_fold_train)
            fold_pred = fold_model.predict(X_fold_val)
            cv_scores.append(balanced_accuracy_score(y_fold_val, fold_pred))

        cv_scores = np.array(cv_scores, dtype=float)
        print(f"CV balanced accuracy: {cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")
    except Exception as e:
        print(f"Cross-validation skipped due to error: {e}")

    # Refit on all available data before saving the final model.
    final_model = build_model(random_state=RANDOM_STATE)
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")

    metadata = {
        "feature_columns": list(X.columns),
        "target_column": "Class",
        "model_file": MODEL_PATH,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "evaluation_metric": "balanced_accuracy",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {os.path.abspath(METADATA_PATH)}")


if __name__ == "__main__":
    main()
