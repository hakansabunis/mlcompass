import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"
TARGET_COLUMN = "Class"
MODEL_PATH = os.path.join(os.getcwd(), "trained_model.joblib")

df = pd.read_csv(CSV_PATH)

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {CSV_PATH}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

categorical_columns = ["V2"]
numeric_columns = [col for col in X.columns if col not in categorical_columns]

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
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ],
    remainder="drop",
)

classifier = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print(f"Model trained and saved to {MODEL_PATH}")
