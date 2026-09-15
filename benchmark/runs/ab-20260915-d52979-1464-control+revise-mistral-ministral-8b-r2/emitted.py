import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump
from pathlib import Path
import os
from tqdm import tqdm

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"

    # Verify file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist")

    # Load the dataset with error handling
    try:
        print("Loading data...")
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {str(e)}")

    # Verify required columns exist
    required_columns = ['V1', 'V2', 'V3', 'V4', 'Class']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")

    # Separate features and target
    X = df[['V1', 'V2', 'V3', 'V4']]
    y = df['Class']

    # Split data into training and validation sets with stratification
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=20260915,
        stratify=y
    )

    # Initialize the model with reasonable defaults
    print("Training model...")
    model = RandomForestClassifier(
        random_state=20260915,
        n_jobs=-1,
        warm_start=False,
        verbose=0
    )

    # Define hyperparameter grid (more focused for better performance)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # Perform grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    # Fit the model (with progress bar)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model on validation set
    print("\nValidation set performance:")
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Save the best model to a file in current directory
    model_filename = "random_forest_model.joblib"
    dump(best_model, model_filename)
    print(f"\nModel saved successfully as {os.path.abspath(model_filename)}")

    return best_model, model_filename

if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
