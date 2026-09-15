import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def train_and_save_model(csv_path, target_column='price', output_model_path='house_price_model.joblib'):
    """
    Train a model to predict house prices and save it to a file.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        target_column (str): Name of the target column (default: 'price').
        output_model_path (str): Path to save the trained model (default: 'house_price_model.joblib').
    """
    # Load the data
    try:
        df = pd.read_csv(csv_path)
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with preprocessing and model
    # We'll use StandardScaler and RandomForestRegressor
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE on test set: {rmse:.2f}")

    # Save the model
    try:
        dump(model, output_model_path)
        print(f"Model saved successfully to {output_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Define the path to the CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r1\train.csv"

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
    else:
        train_and_save_model(csv_path)
