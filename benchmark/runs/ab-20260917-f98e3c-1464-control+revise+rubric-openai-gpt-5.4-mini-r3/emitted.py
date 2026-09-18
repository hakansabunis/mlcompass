import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"
MODEL_PATH = "trained_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found.")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def build_models(random_state: int = 42):
    return {
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "et": ExtraTreesClassifier(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=None,
            max_iter=300,
            random_state=random_state,
        ),
        "lr": LogisticRegression(
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def make_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def main():
    df = load_data(DATA_PATH)
    df = deduplicate(df)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Hold-out split for unbiased evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scored_models = []
    for name, model in models.items():
        pipe = make_pipeline(model)
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        scored_models.append((name, float(np.mean(scores)), float(np.std(scores))))
        print(f"{name}: balanced_accuracy={np.mean(scores):.6f} +/- {np.std(scores):.6f}")

    scored_models.sort(key=lambda x: x[1], reverse=True)
    best_name = scored_models[0][0]
    best_model = models[best_name]
    final_pipeline = make_pipeline(best_model)

    # Fit only on training data
    final_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test data
    test_pred = final_pipeline.predict(X_test)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Best model: {best_name}")
    print(f"Held-out test balanced accuracy: {test_bal_acc:.6f}")
    print(f"Held-out test accuracy: {test_acc:.6f}")

    artifact = {
        "model": final_pipeline,
        "feature_names": list(X.columns),
        "target_name": "Class",
        "best_model_name": best_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_results": [
            {"model": n, "mean_balanced_accuracy": m, "std_balanced_accuracy": s}
            for n, m, s in scored_models
        ],
        "held_out_test_metrics": {
            "balanced_accuracy": float(test_bal_acc),
            "accuracy": float(test_acc),
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
