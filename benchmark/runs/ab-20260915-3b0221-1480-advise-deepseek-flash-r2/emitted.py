import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"

df = pd.read_csv(train_path)

target_col = "Class"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = [
    col for col in X.columns
    if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == object
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X, y)

joblib.dump(model, "model.joblib")
