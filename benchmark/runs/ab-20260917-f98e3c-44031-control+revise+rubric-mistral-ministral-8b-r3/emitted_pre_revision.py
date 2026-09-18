import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed2-r3\train.csv"

    # Load the data
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.2f}")

    # Save the trained model to current directory
    model_path = os.path.join(os.getcwd(), 'house_price_model.joblib')
    dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
