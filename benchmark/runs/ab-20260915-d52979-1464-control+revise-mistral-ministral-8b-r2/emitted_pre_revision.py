import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump
from pathlib import Path

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"

    # Load the dataset
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Initialize the model with hyperparameters to tune
    model = RandomForestClassifier(random_state=20260915)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform grid search with cross-validation
    print("Training model with hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model on validation set
    print("\nValidation set performance:")
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Save the best model to a file
    model_filename = "random_forest_model.joblib"
    dump(best_model, model_filename)
    print(f"\nModel saved as {model_filename}")

    return best_model, model_filename

if __name__ == "__main__":
    train_and_save_model()
