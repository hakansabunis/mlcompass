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
    # Define file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"

    # Load the dataset
    try:
        data = pd.read_csv(csv_path)
        print("Dataset loaded successfully with shape:", data.shape)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915
    )

    # Create a preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create a full pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=20260915,
            max_depth=10,
            min_samples_split=5
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained and evaluated. Test MSE: {mse:.2f}")

    # Save the trained model
    model_filename = "house_price_model.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {os.path.abspath(model_filename)}")

if __name__ == "__main__":
    train_and_save_model()
