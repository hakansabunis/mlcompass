import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with scaling and RandomForestRegressor
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Test MSE: {mse:.2f}")

    # Save the model to current directory
    model_filename = 'house_price_model.joblib'
    dump(model, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
