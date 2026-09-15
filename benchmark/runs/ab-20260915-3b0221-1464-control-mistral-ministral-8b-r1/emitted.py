import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import os

def train_and_save_model(csv_path, output_model_path='random_forest_model.joblib'):
    """
    Loads data from CSV, trains a RandomForestClassifier, and saves the model.

    Args:
        csv_path (str): Path to the CSV file
        output_model_path (str): Path to save the trained model
    """
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=20260915,
        class_weight='balanced',  # handles potential class imbalance
        n_jobs=-1  # use all available cores
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained and evaluated. Test accuracy: {accuracy:.4f}")

    # Save the model
    dump(model, output_model_path)
    print(f"Model saved to {os.path.abspath(output_model_path)}")

if __name__ == "__main__":
    # Define the path to your CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"

    # Ensure the directory exists for the output model
    os.makedirs(os.path.dirname('random_forest_model.joblib'), exist_ok=True)

    # Train and save the model
    train_and_save_model(csv_path)
