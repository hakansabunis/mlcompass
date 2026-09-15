import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os
import warnings

def train_and_save_model():
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Define file path with raw string for Windows path handling
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r1\train.csv"

    try:
        # Load the data with explicit dtype handling to prevent potential issues
        data = pd.read_csv(file_path, dtype={
            'MedInc': np.float64,
            'HouseAge': np.float64,
            'AveRooms': np.float64,
            'AveBedrms': np.float64,
            'Population': np.float64,
            'AveOccup': np.float64,
            'Latitude': np.float64,
            'Longitude': np.float64,
            'price': np.float64
        })

        # Check for missing values
        if data.isnull().sum().any():
            print("Warning: The dataset contains missing values. Handling strategy not implemented in this script.")
            return

        # Separate features and target
        X = data.drop(columns=['price'])
        y = data['price']

        # Split data into training and validation sets with stratification
        # (Note: Stratification isn't directly applicable for regression, but random_state ensures reproducibility)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=20260915
        )

        # Initialize and fit a preprocessing pipeline (scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Initialize and train a Gradient Boosting Regressor with more robust parameters
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=20260915,
            verbose=1
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        val_predictions = model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        print(f"Validation RMSE: {val_rmse:.4f}")

        # Save the model and scaler to disk in a single joblib file
        model_filename = "house_price_model.joblib"

        # Create a dictionary to save both model and scaler
        saved_objects = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }

        dump(saved_objects, model_filename)

        print(f"Model and preprocessing objects saved to {os.path.abspath(model_filename)}")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()
