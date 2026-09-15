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
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"

    # Load the data
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with standardization and RandomForest
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=2
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.2f}")

    # Save the trained model
    model_filename = 'house_price_model.joblib'
    dump(model, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
