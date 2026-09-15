import os
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Build a robust classifier pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Fit on the full dataset
    model.fit(X, y)

    # Save fitted model and metadata
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
