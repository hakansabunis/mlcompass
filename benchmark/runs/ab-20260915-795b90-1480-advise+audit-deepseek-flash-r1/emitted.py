import random
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# Set global random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
TARGET_COL = "Class"
MODEL_FILE = Path.cwd() / "model.joblib"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    # Candidate models
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=SEED
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=SEED
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    }

    # Hold out a validation split for model selection / evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        mean_score = scores.mean()
        print(f"{name}: CV accuracy = {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    print(f"Best model by CV: {best_name} (CV accuracy = {best_score:.4f})")

    # Evaluate best pipeline on the held-out validation set
    best_pipeline.fit(X_train, y_train)
    val_preds = best_pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy for {best_name}: {val_acc:.4f}")

    # Refit best pipeline on the full dataset
    best_pipeline.fit(X, y)

    # Save model
    joblib.dump(best_pipeline, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
