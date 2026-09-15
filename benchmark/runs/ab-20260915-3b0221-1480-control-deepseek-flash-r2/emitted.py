import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Path to the training CSV
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
target_col = "Class"

# Load data
df = pd.read_csv(csv_path)

X = df.drop(columns=[target_col])
y = df[target_col]

# Column groups
numeric_cols = ["V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]
categorical_cols = ["V2"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Classifier
classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# Optional validation split (safe for small/rare classes)
stratify_arg = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Validation accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Refit on the full training data before saving
model.fit(X, y)

# Save the fitted model to the current working directory
model_path = os.path.join(os.getcwd(), "model.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
