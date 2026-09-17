import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in the dataset.")

    # Split features/target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Basic train/validation split for a quick sanity check
    stratify = y if y.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    # Simple, robust model pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Fit model
    model.fit(X_train, y_train)

    # Optional validation metric
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc:.6f}")

    # Refit on full dataset before saving
    model.fit(X, y)

    # Save fitted model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
