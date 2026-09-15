import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os
from tqdm import tqdm

def load_and_preprocess_data(filepath):
    """Load and preprocess the data, handling potential missing values."""
    try:
        data = pd.read_csv(filepath)

        # Check for missing values and handle them
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        if data.empty:
            raise ValueError("Data is empty after handling missing values")

        # Separate features and target
        X = data.drop('price', axis=1)
        y = data['price']

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_val, y_train, y_val

    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        raise

def train_and_evaluate_model(X_train, X_val, y_train, y_val):
    """Train a RandomForestRegressor and evaluate its performance."""
    try:
        # Initialize and train the model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1  # Use all available cores for faster training
        )

        # Fit the model with progress bar
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Model RMSE on validation set: {rmse:.2f}")

        return model

    except Exception as e:
        print(f"Error in model training: {e}")
        raise

def save_model(model, output_path):
    """Save the trained model to a file."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the model
        dump(model, output_path)
        print(f"Model saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    """Main function to orchestrate the workflow."""
    try:
        # Define file paths
        csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed2-r3\train.csv"
        model_output_path = os.path.join(os.getcwd(), "house_price_model.joblib")

        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_train, X_val, y_train, y_val = load_and_preprocess_data(csv_path)

        # Train and evaluate model
        print("\nTraining model...")
        model = train_and_evaluate_model(X_train, X_val, y_train, y_val)

        # Save the model
        print("\nSaving model...")
        save_model(model, model_output_path)

    except Exception as e:
        print(f"An error occurred in the main workflow: {e}")

if __name__ == "__main__":
    main()
