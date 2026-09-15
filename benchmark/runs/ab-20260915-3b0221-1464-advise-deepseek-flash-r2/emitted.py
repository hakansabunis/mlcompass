import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
CSV_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"
MODEL_PATH = os.path.join(os.getcwd(), "model.joblib")

def main():
    df = pd.read_csv(CSV_PATH)

    # Use de-duplicated data for model selection to avoid inflated CV scores
    df_unique = df.drop_duplicates()
    X_cv = df_unique.drop(columns=["Class"])
    y_cv = df_unique["Class"].astype(int)

    X_full = df.drop(columns=["Class"])
    y_full = df["Class"].astype(int)

    candidates = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "rf": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        "et": make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(
                n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        "gb": make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
        "hgb": make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
        "svc": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE),
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in candidates.items():
        scores = cross_val_score(
            model, X_cv, y_cv, cv=cv, scoring="accuracy"
        )
        mean_score = scores.mean()
        print(f"{name}: {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = model

    print(f"Best model: {best_name} with CV accuracy {best_score:.4f}")
    best_model.fit(X_full, y_full)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
