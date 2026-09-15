import os
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
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
OUTPUT_MODEL = "model.joblib"
TARGET_COL = "Class"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

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

    models = {
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(random_state=42),
        "hgb": HistGradientBoostingClassifier(random_state=42),
        "lr": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "svc": SVC(
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_score = -np.inf
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mean_score = float(np.mean(scores))
        print(f"{name}: roc_auc = {mean_score:.4f} (+/- {np.std(scores):.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    print(f"Best model: {best_name} with roc_auc = {best_score:.4f}")

    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, OUTPUT_MODEL)
    print(f"Saved fitted model to {os.path.abspath(OUTPUT_MODEL)}")


if __name__ == "__main__":
    main()
