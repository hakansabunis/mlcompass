import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed2-r3\train.csv"
target_col = "Class"

df = pd.read_csv(csv_path)

X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = ["V2"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X, y)

output_path = os.path.join(os.getcwd(), "model.joblib")
joblib.dump(model, output_path)
