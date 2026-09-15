import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm

def train_and_save_model(csv_path, model_save_path='model.joblib'):
    """
    Load data, preprocess, train a model, and save it to disk.

    Args:
        csv_path (str): Path to the CSV file containing the data
        model_save_path (str): Path where the trained model will be saved
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', pd.get_dummies(columns=[c for c in categorical_cols]))
    ])

    # Combine preprocessing steps
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Create a pipeline with preprocessing and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=20260915,
            n_jobs=-1,
            verbose=1
        ))
    ])

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_score = model.score(X_val, y_val)
    print(f"\nModel validation accuracy: {val_score:.4f}")

    # Save the trained model
    dump(model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv"
    train_and_save_model(csv_path)
