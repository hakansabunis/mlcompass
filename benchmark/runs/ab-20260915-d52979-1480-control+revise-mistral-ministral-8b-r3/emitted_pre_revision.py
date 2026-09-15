import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm

# Define file path
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(csv_path)

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=20260915, stratify=y)

# Define preprocessing for numerical and categorical columns
numerical_cols = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
categorical_cols = ['V2']

# Create transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=20260915))
])

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate on validation set
val_score = model.score(X_val, y_val)
print(f"Validation accuracy: {val_score:.4f}")

# Save the model to a file
model_path = "trained_model.joblib"
dump(model, model_path)
print(f"Model saved to {model_path}")
