import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
import os

def train_and_save_model():
    # Define file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed2-r3\train.csv"

    # Load the dataset
    try:
        data = pd.read_csv(csv_path)
        print("Data loaded successfully. Shape:", data.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Convert categorical feature to numerical (V2)
    X['V2'] = X['V2'].astype('category').cat.codes

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save the trained model
    model_file = os.path.join(os.getcwd(), 'random_forest_model.joblib')
    dump(model, model_file)
    print(f"\nModel saved to: {model_file}")

if __name__ == "__main__":
    train_and_save_model()
