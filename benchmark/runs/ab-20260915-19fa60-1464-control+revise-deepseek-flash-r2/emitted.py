import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import joblib

def main():
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"
    target_col = "Class"

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    if y.nunique() < 2:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), X.columns.tolist())
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X, y)

    output_path = Path.cwd() / "model.joblib"
    joblib.dump(model, output_path)
    print(f"Saved fitted model to: {output_path}")

if __name__ == "__main__":
    main()
