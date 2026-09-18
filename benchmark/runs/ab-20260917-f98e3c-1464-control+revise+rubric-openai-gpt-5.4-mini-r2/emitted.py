import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier

TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
RANDOM_STATE = 42


def load_data(path: str):
    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in training data.")

    # Remove duplicate rows before splitting
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def build_model(n_classes: int, random_state: int = RANDOM_STATE):
    if n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    feature_cols = ["V1", "V2", "V3", "V4"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe, feature_cols


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)

    X, y = load_data(TRAIN_PATH)
    n_classes = y.nunique()

    pipe, feature_cols = build_model(n_classes=n_classes, random_state=RANDOM_STATE)

    # Hold out a test split for unbiased evaluation
    stratify = y if n_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # Cross-validation on the training split only for a more robust estimate
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        print(f"CV balanced accuracy: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
    except Exception as e:
        print(f"Cross-validation skipped due to: {e}")

    # Fit on training split only
    pipe.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Held-out test accuracy: {test_acc:.5f}")
    print(f"Held-out test balanced accuracy: {test_bal_acc:.5f}")

    # Save final model fitted on all available data for later use
    pipe.fit(X, y)
    joblib.dump(
        {
            "model": pipe,
            "feature_columns": feature_cols,
            "target_column": "Class",
            "classes_": y.unique().tolist(),
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )

    print(f"Saved trained model to: {os.path.abspath(MODEL_PATH)}")
