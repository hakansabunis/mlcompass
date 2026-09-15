import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Check for missing values
    if X.isnull().values.any():
        print("Warning: Missing values detected. Filling with median.")
        X = X.fillna(X.median())

    return X, y

def train_model(X, y):
    """Train a model pipeline with hyperparameter tuning."""
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Define hyperparameter grid for tuning
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print("Starting model training...")
    grid_search.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = grid_search.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation R²: {r2:.2f}")

    return grid_search.best_estimator_

def save_model(model, model_path='house_price_model.joblib'):
    """Save the trained model to a file."""
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    # Define file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)

    # Train model
    model = train_model(X, y)

    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
