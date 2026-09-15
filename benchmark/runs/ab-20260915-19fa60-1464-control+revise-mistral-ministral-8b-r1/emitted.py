import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os
from tqdm import tqdm

def main():
    # Define file path with raw string to handle backslashes properly
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"

    try:
        # Load data with error handling
        print("Loading data...")
        data = pd.read_csv(csv_path)

        # Verify required columns exist
        required_columns = ['V1', 'V2', 'V3', 'V4', 'Class']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

        # Separate features and target
        X = data[['V1', 'V2', 'V3', 'V4']].values  # Convert to numpy array for better performance
        y = data['Class'].values

        # Split data with stratification and progress tracking
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
            shuffle=True
        )

        # Initialize and train model with progress tracking
        print("Training model...")
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,  # Use all available cores
            verbose=1   # Show progress
        )

        # Train the model
        clf.fit(X_train, y_train)

        # Evaluate on test set
        print("Evaluating model...")
        y_pred = clf.predict(X_test)

        # Calculate and print metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save the trained model with progress
        model_filename = "trained_model.joblib"
        print(f"\nSaving model to {model_filename}...")
        dump(clf, model_filename, compress=3)  # High compression
        print(f"Model saved successfully at: {os.path.abspath(model_filename)}")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
