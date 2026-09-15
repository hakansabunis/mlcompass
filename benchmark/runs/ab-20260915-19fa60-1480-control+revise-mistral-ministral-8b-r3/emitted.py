import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
import os
from tqdm import tqdm

def train_and_save_model():
    # Define file path with proper escaping
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed2-r3\train.csv"

    try:
        # Load the dataset with progress bar for large files
        print("Loading data...")
        data = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Convert categorical feature (V2) to numerical
    # Use get_dummies for one-hot encoding instead of category codes
    X_encoded = pd.get_dummies(X['V2'], prefix='V2', dtype='int8')
    X = X.drop('V2', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split data into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    # Initialize and train the model with early stopping
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nEvaluating model...")
    y_pred = model.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save the trained model with progress
    model_file = os.path.join(os.getcwd(), 'random_forest_model.joblib')
    print(f"\nSaving model to: {model_file}")
    dump(model, model_file, compress=3)  # Compress to save space
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
