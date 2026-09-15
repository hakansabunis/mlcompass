import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os
import warnings

def train_and_save_model():
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Define the file path with proper path handling
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"

    try:
        # Load the data with error handling
        data = pd.read_csv(csv_path)

        # Check for missing values
        if data.isnull().sum().any().any():
            print("Warning: Missing values found. Using simple imputation.")
            data = data.fillna(data.mean())

        # Separate features and target
        X = data.drop(columns=['price'])
        y = data['price']

        # Check for infinite values
        if np.isinf(X).any().any() or np.isinf(y).any():
            print("Warning: Infinite values found. Removing them.")
            X = X.replace([np.inf, -np.inf], np.nan)
            y = y.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            y = y.dropna()

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create a more robust pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1  # Use all available cores
            ))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        val_preds = model.predict(X_val)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        r2 = r2_score(y_val, val_preds)

        # Print evaluation metrics
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R² Score: {r2:.4f}")

        # Get feature importances
        importances = model.named_steps['regressor'].feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\nTop 5 Important Features:")
        print(feature_importances.head())

        # Save the trained model
        model_filename = 'house_price_model.joblib'
        dump(model, model_filename)
        print(f"\nModel successfully saved as {os.path.abspath(model_filename)}")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    train_and_save_model()
