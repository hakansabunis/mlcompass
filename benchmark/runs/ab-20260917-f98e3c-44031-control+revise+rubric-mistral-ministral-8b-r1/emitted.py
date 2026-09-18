import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the data from CSV file."""
    # Load the data
    df = pd.read_csv(filepath)

    # Remove duplicate rows before any processing
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Standardize the features (only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a Gradient Boosting Regressor model."""
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Test MSE: {mse:.2f}")
    return y_pred

def save_model(model, scaler, output_path='model.joblib'):
    """Save the trained model and scaler to a file."""
    # Save both model and scaler in a single file
    dump({'model': model, 'scaler': scaler}, output_path)
    print(f"Model and scaler saved to {output_path}")

def main():
    # File path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model on unseen data (test set)
    evaluate_model(model, X_test, y_test)

    # Save the model and scaler
    save_model(model, scaler)

if __name__ == "__main__":
    main()
