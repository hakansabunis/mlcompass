import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os
import warnings

def train_and_save_model():
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Define the file path with proper escaping for Windows
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"

    try:
        # Check if file exists before proceeding
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The file {csv_path} does not exist.")

        # Load the dataset with error handling
        print("Loading dataset...")
        data = pd.read_csv(csv_path)

        # Check for missing values
        if data.isnull().sum().any().any():
            print("Warning: Missing values detected in the dataset. Handling with imputation.")
            data = data.fillna(data.median())  # Simple median imputation

        # Separate features and target
        X = data.drop('price', axis=1)
        y = data['price']

        # Handle potential infinite values that might have been introduced
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        # Split data into train and test sets with stratification (though not ideal for regression)
        # Using random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=20260915  # Using seed from path
        )

        # Create a pipeline with scaling and RandomForestRegressor
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=200,  # Increased from 100 for better performance
                max_depth=None,     # Let the tree grow naturally
                min_samples_split=5,
                random_state=20260915,  # Using seed from path
                n_jobs=-1,
                verbose=1  # Show progress
            ))
        ])

        # Train the model
        print("\nTraining model...")
        model.fit(X_train, y_train)

        # Evaluate the model with multiple metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\nModel evaluation on test set:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

        # Save the model to current directory with consistent filename
        model_filename = 'house_price_regressor.joblib'
        dump(model, model_filename)
        print(f"\nModel successfully saved as '{model_filename}' in the current directory.")

    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()
