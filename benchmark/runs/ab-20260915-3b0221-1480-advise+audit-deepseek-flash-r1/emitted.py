import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = Path(r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv")
MODEL_PATH = Path.cwd() / "model.joblib"

TARGET_COL = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    categorical_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    best_name, best_score, best_model = None, -np.inf, None
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        acc = accuracy_score(y_val, preds)
        print(f"{name}: macro_f1={f1:.4f} acc={acc:.4f}")
        if f1 > best_score:
            best_name, best_score, best_model = name, f1, pipe

    # Refit the best pipeline on all data
    final_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", candidates[best_name])])
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_PATH)
    print(f"Best model: {best_name} (val macro_f1={best_score:.4f})")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
