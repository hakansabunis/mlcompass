import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
MODEL_PATH = "model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = ["V2"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)
    dump(pipeline, MODEL_PATH)

if __name__ == "__main__":
    main()
