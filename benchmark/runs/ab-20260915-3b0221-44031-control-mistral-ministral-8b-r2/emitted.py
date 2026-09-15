import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def train_and_save_model(csv_path, target_column='price'):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=20260915,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained with MSE: {mse:.2f}")

    # Save the model and scaler
    model_filename = 'house_price_model.joblib'
    scaler_filename = 'house_price_scaler.joblib'

    dump(model, model_filename)
    dump(scaler, scaler_filename)

    print(f"Model saved as '{model_filename}'")
    print(f"Scaler saved as '{scaler_filename}'")

if __name__ == "__main__":
    # Define the path to the CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file at {csv_path} was not found.")

    # Train and save the model
    train_and_save_model(csv_path)
