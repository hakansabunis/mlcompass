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
    """Load and preprocess the data from CSV file."""
    try:
        # Load data with explicit dtype conversion to handle potential issues
        dtypes = {
            'V1': 'int64',
            'V2': 'category',  # Treat as categorical for one-hot encoding
            'V3': 'float64',
            'V4': 'float64',
            'V5': 'int64',
            'V6': 'int64',
            'V7': 'int64',
            'V8': 'float64',
            'V9': 'float64',
            'V10': 'float64',
            'Class': 'int64'
        }
        df = pd.read_csv(filepath, dtype=dtypes)

        # Separate features and target
        X = df.drop(columns=['Class'])
        y = df['Class']

        # Convert categorical feature (V2) to numerical using one-hot encoding
        # Only drop first category if needed, but keep all for better model performance
        X = pd.get_dummies(X, columns=['V2'], drop_first=False)

        return X, y

    except Exception as e:
        raise RuntimeError(f"Error loading or preprocessing data: {str(e)}")

def train_and_evaluate_model(X, y):
    """Train a RandomForestClassifier and evaluate it."""
    try:
        # Split data into train and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=20260915, stratify=y
        )

        # Initialize and train the model with reasonable parameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=20260915,
            n_jobs=-1,
            verbose=1,
            class_weight='balanced'  # Handle potential class imbalance
        )

        # Train the model with progress tracking
        model.fit(X_train, y_train)

        # Make predictions on validation set
        y_pred = model.predict(X_val)

        # Calculate and print evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        return model, X_train, X_val, y_train, y_val

    except Exception as e:
        raise RuntimeError(f"Error during model training/evaluation: {str(e)}")

def save_model(model, filepath='trained_model.joblib'):
    """Save the trained model to a file in current working directory."""
    try:
        dump(model, filepath)
        print(f"\nModel successfully saved to {os.path.abspath(filepath)}")
        return True
    except Exception as e:
        raise RuntimeError(f"Error saving model: {str(e)}")

def main():
    # Define file path with raw string to avoid escape character issues
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv"

    # Verify file existence before proceeding
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The specified file does not exist: {csv_path}")

    try:
        print("Starting data processing and model training...")
        print(f"Loading data from: {csv_path}")

        # Load and preprocess data
        X, y = load_and_preprocess_data(csv_path)

        # Train and evaluate model
        model, _, _, _, _ = train_and_evaluate_model(X, y)

        # Save the model
        save_model(model)

        print("\nTraining process completed successfully.")

    except Exception as e:
        print(f"\nError during main execution: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    main()
