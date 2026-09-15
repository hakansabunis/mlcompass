import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from pathlib import Path

def train_and_save_model():
    # Define the file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"

    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Initialize and train the model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=20260915,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate on validation set
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Save the model to current working directory
    model_path = Path.cwd() / "random_forest_model.joblib"
    print(f"Saving model to {model_path}")
    dump(model, model_path)

    return model_path

if __name__ == "__main__":
    train_and_save_model()
