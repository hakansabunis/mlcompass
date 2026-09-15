import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------------
# Reproducibility: fix all relevant seeds.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
OUTPUT_MODEL = "model.joblib"
TARGET_COL = "Class"


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "str"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ------------------------------------------------------------------
    # Hold-out validation split (stratified to preserve class balance).
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=SEED,
        stratify=y,
    )
    print(f"Train rows: {len(X_train)}, Validation rows: {len(X_val)}")

    preprocessor = build_preprocessor(X_train)

    models = {
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(random_state=SEED),
        "hgb": HistGradientBoostingClassifier(random_state=SEED),
        "lr": LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            class_weight="balanced",
        ),
        "svc": SVC(
            probability=True,
            random_state=SEED,
            class_weight="balanced",
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_score = -np.inf
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mean_score = float(np.mean(scores))
        print(f"{name}: CV roc_auc = {mean_score:.4f} (+/- {np.std(scores):.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    print(f"Best model by CV: {best_name} (roc_auc = {best_score:.4f})")

    # ------------------------------------------------------------------
    # Fit best model on training data and evaluate on held-out validation.
    # ------------------------------------------------------------------
    best_pipeline.fit(X_train, y_train)

    val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    val_pred = best_pipeline.predict(X_val)
    print(f"Validation roc_auc : {roc_auc_score(y_val, val_proba):.4f}")
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")

    # ------------------------------------------------------------------
    # Refit on the full dataset before persisting the final model.
    # ------------------------------------------------------------------
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", models[best_name]),
        ]
    )
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, OUTPUT_MODEL)
    print(f"Saved fitted model to {os.path.abspath(OUTPUT_MODEL)}")


if __name__ == "__main__":
    main()
