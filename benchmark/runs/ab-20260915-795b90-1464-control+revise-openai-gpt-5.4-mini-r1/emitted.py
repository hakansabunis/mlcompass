import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
    model_path = "trained_model.joblib"
    target_col = "Class"

    # Load data
    df = pd.read_csv(train_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols]
    y = df[target_col]

    if y.nunique(dropna=False) < 2:
        raise ValueError("Target column must contain at least two classes.")

    # Model pipeline
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    # Optional cross-validation for a quick sanity check
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            print(f"CV accuracy: {scores.mean():.6f} ± {scores.std():.6f}")

    # Fit on full dataset
    model.fit(X, y)

    # Save the trained model and metadata
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(bundle, model_path)

    print(f"Saved model to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()
