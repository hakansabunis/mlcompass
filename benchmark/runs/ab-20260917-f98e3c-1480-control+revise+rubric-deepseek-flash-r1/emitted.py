import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

# Set global random seed for reproducibility
np.random.seed(42)

# Path to the training CSV
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"

# Load the data
df = pd.read_csv(train_path)

# Remove duplicate rows before any split
df = df.drop_duplicates()

# Drop rows with missing target
df = df.dropna(subset=["Class"])

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the full pipeline with a classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])

# Split data into train and test sets for evaluation
stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# Train on the training set only
pipeline.fit(X_train, y_train)

# Evaluate on the held-out test set
y_pred = pipeline.predict(X_test)
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))

# Refit on the full dataset for the final saved model
pipeline.fit(X, y)

# Save the fitted model to the current working directory
joblib.dump(pipeline, "model.joblib")
