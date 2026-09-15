import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    try:
        print("Loading dataset...")
        data = pd.read_csv(filepath)

        # Check for missing values
        if data.isnull().sum().sum() > 0:
            print("Warning: Dataset contains missing values. Imputing with median.")
            data = data.fillna(data.median())

        # Separate features and target
        X = data.drop(columns=['Class'])
        y = data['Class']

        # Convert to numpy arrays for sklearn compatibility
        X = X.values
        y = y.values

        return X, y

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train_model(X, y):
    """Train and evaluate the model"""
    try:
        # Split into train and validation sets
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=20260915,  # Using seed from path
            stratify=y
        )

        # Initialize and train the model
        print("Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=20260915,  # Using seed from path
            class_weight='balanced',
            max_depth=10,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)

        # Evaluate on validation set
        print("Evaluating model...")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        print(f"Validation accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        return model

    except Exception as e:
        print(f"Error during training: {e}")
        raise

def save_model(model, filename):
    """Save the trained model"""
    try:
        dump(model, filename)
        print(f"Model successfully saved to {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    # Define file path from the given location
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"

    # Verify file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(csv_path)

        # Train model
        model = train_model(X, y)

        # Save model
        save_model(model, "mlcab_model.joblib")

        print("\nTraining pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
