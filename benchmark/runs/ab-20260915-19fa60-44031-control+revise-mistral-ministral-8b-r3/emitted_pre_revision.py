import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)

    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val

def train_and_evaluate_model(X_train, X_val, y_train, y_val):
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Model RMSE on validation set: {rmse:.2f}")

    return model

def save_model(model, output_path):
    # Save the model to a file
    dump(model, output_path)
    print(f"Model saved to {output_path}")

def main():
    # Define file paths
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed2-r3\train.csv"
    model_output_path = os.path.join(os.getcwd(), "house_price_model.joblib")

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(csv_path)

    # Train and evaluate model
    model = train_and_evaluate_model(X_train, X_val, y_train, y_val)

    # Save the model
    save_model(model, model_output_path)

if __name__ == "__main__":
    main()
