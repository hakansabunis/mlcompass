import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from pathlib import Path

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    # Load data with error handling for missing values
    df = pd.read_csv(filepath)

    # Remove duplicate rows before splitting
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into train and test sets (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Only fit on training data
    X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test data

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a RandomForest classifier with class imbalance handling"""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  # Handles class imbalance
        max_depth=10,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with metrics that work with imbalanced data"""
    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def save_model(model, scaler, model_path='trained_model.joblib'):
    """Save the trained model and scaler to disk"""
    dump({
        'model': model,
        'scaler': scaler
    }, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    # Define file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, scaler)

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
