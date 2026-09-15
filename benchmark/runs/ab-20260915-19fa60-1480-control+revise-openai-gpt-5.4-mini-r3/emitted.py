import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed2-r3\train.csv"
MODEL_PATH = "trained_model.joblib"
RANDOM_STATE = 42


def make_one_hot_encoder():
    # Compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_cols),
            ("cat", categorical_preprocess, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess, categorical_cols, numeric_cols


def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    preprocess, categorical_cols, numeric_cols = build_preprocessor(X)

    candidates = []

    candidates.append(
        (
            "logreg",
            Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        )
    )

    candidates.append(
        (
            "hgb",
            Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=None,
                            max_iter=300,
                            min_samples_leaf=20,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        )
    )

    best_name = None
    best_score = -np.inf
    best_model = None

    # Cross-validate if possible; otherwise fit the first candidate
    n_classes = y.nunique()
    if n_classes > 1:
        class_counts = y.value_counts()
        cv = min(5, int(class_counts.min()))
        if cv >= 2:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            for name, model in candidates:
                try:
                    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
                    score = float(np.mean(scores))
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_model = model
                except Exception:
                    continue

    if best_model is None:
        best_name, best_model = candidates[0]
        best_score = None

    best_model.fit(X, y)

    artifact = {
        "model": best_model,
        "metadata": {
            "target_column": target_col,
            "feature_columns": list(X.columns),
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
            "selected_model": best_name,
            "cv_accuracy": best_score,
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
