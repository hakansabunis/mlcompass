import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm

def train_and_save_model(csv_path, model_save_path='model.joblib'):
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Identify column types for preprocessing
    numeric_features = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    categorical_features = ['V2']

    # Create transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle outliers with median
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=20260915,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    # Train the model
    print("Training model...")
    model.fit(X, y)

    # Save the model
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"
    train_and_save_model(csv_path)
