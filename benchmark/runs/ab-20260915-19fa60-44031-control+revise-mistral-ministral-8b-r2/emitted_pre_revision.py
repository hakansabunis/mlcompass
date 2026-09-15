import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed1-r2\train.csv"

    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return

    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Split the data into training and temporary validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split the temporary validation set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Create a preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values if any
        ('scaler', StandardScaler())  # Standardize features
    ])

    # Create a pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {val_rmse:.2f}")

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Test RMSE: {test_rmse:.2f}")

    # Save the trained model
    model_filename = os.path.join(os.getcwd(), 'house_price_model.joblib')
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
