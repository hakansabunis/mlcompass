import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

CSV_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
TARGET_COL = "price"
MODEL_PATH = os.path.join(os.getcwd(), "trained_model.joblib")


def main():
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=600,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                    early_stopping=False,
                ),
            ),
        ]
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
