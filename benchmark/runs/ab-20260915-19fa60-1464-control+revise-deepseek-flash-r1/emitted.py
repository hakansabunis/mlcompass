import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

INPUT_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
OUTPUT_PATH = os.path.join(os.getcwd(), "model.joblib")


def main():
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Training file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Normalize column names to guard against stray whitespace in headers.
    df.columns = [str(c).strip() for c in df.columns]

    if "Class" not in df.columns:
        raise ValueError(
            f"Target column 'Class' not found. Available columns: {list(df.columns)}"
        )

    # Drop rows without a label; keep everything else.
    df = df.dropna(subset=["Class"])

    if df.empty:
        raise ValueError("No usable rows after dropping missing target values.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if X.shape[1] == 0:
        raise ValueError("No feature columns available for training.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved trained model to: {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
