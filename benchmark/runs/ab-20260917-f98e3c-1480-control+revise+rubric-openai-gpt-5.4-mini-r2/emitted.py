import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"
MODEL_PATH = "trained_model.joblib"
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(TRAIN_PATH)

    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data.")

    # Remove duplicate rows before splitting to avoid leakage from repeated examples.
    df = df.drop_duplicates().reset_index(drop=True)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Train/test split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y if y.nunique() > 1 else None,
    )

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

    # Model
    is_multiclass = y.nunique() > 2
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="multi:softprob" if is_multiclass else "binary:logistic",
        eval_metric="mlogloss" if is_multiclass else "logloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Optional cross-validation on training split only
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            cv_scores.append(balanced_accuracy_score(y_val, preds))
        print(f"CV balanced accuracy: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}")
    except Exception as e:
        print(f"Cross-validation skipped due to error: {e}")

    # Fit on training split only
    clf.fit(X_train, y_train)

    # Evaluate on held-out test split
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)
    test_f1_macro = f1_score(y_test, test_preds, average="macro")

    print(f"Holdout accuracy: {test_acc:.5f}")
    print(f"Holdout balanced accuracy: {test_bal_acc:.5f}")
    print(f"Holdout macro F1: {test_f1_macro:.5f}")

    # Save model artifact
    artifact = {
        "model": clf,
        "feature_columns": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_col": target_col,
        "classes_": sorted(y.unique().tolist()),
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
