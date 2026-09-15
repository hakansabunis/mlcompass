import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


TRAIN_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def build_model():
    numeric_features = ["V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]
    categorical_features = ["V2"]

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    df = pd.read_csv(TRAIN_PATH)

    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Ensure categorical column is treated as string/object consistently
    if "V2" in X.columns:
        X["V2"] = X["V2"].astype("string")

    clf = build_model()

    # Optional cross-validation for a quick sanity check
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"CV accuracy: {scores.mean():.5f} +/- {scores.std():.5f}")
    except Exception as e:
        print(f"Cross-validation skipped due to: {e}")

    # Fit on full training data
    clf.fit(X, y)

    artifact = {
        "model": clf,
        "features": list(X.columns),
        "target": target_col,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
