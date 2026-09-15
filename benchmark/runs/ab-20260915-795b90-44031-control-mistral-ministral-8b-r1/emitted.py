import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm import tqdm

def train_and_save_model():
    # Define file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"

    # Load the dataset
    print("Loading data...")
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize and train the model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\nValidation RMSE: {rmse:.2f}")

    # Save the model and scaler
    model_filename = "house_price_model.joblib"
    scaler_filename = "house_price_scaler.joblib"

    dump(model, model_filename)
    dump(scaler, scaler_filename)

    print(f"\nModel and scaler saved to {model_filename} and {scaler_filename}")

if __name__ == "__main__":
    train_and_save_model()
