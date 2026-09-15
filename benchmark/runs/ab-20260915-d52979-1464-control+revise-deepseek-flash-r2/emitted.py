import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier


def main():
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"
    target_col = "Class"

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if len(y) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        out_path = os.path.join(os.getcwd(), "model.joblib")
        joblib.dump(model, out_path)
        print(f"Not enough samples for CV. Saved fallback model to {out_path}")
        return

    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count >= 2:
        n_splits = min(5, min_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = max(2, min(5, len(y)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    candidates = []

    candidates.append((
        "logreg",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],
        },
    ))

    candidates.append((
        "svc",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42)),
        ]),
        {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
            "model__class_weight": [None, "balanced"],
        },
    ))

    candidates.append((
        "random_forest",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
        ]),
        {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
    ))

    candidates.append((
        "extra_trees",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(random_state=42, n_jobs=1)),
        ]),
        {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
    ))

    candidates.append((
        "gradient_boosting",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [1, 2],
        },
    ))

    candidates.append((
        "hist_gradient_boosting",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]),
        {
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [15, 31],
        },
    ))

    candidates.append((
        "knn",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    ))

    candidates.append((
        "decision_tree",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]),
        {
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["gini", "entropy"],
            "model__class_weight": [None, "balanced"],
        },
    ))

    candidates.append((
        "lda",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]),
        {},
    ))

    candidates.append((
        "qda",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", QuadraticDiscriminantAnalysis()),
        ]),
        {
            "model__reg_param": [0.0, 0.1, 0.5],
        },
    ))

    candidates.append((
        "gaussian_nb",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
    ))

    best_model = None
    best_score = -np.inf
    best_name = None

    for name, pipeline, param_grid in candidates:
        try:
            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score=np.nan,
            )
            search.fit(X, y)
            score = search.best_score_
            print(f"{name}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = search.best_estimator_
                best_name = name
        except Exception as exc:
            print(f"{name} failed: {exc}")

    if best_model is None:
        print("All candidates failed. Using fallback RandomForest.")
        best_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=500, random_state=42)),
        ])
        best_model.fit(X, y)
        best_name = "fallback_random_forest"
        best_score = float("nan")

    print(f"Best model: {best_name} (CV f1_macro: {best_score:.4f})")

    out_path = os.path.join(os.getcwd(), "model.joblib")
    joblib.dump(best_model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
