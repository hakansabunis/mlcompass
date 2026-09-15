import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.metrics import classification_report

# Set all random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def train_and_save_model(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Remove duplicate rows to avoid inflated performance metrics
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline with scaling and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=5  # Added to prevent overfitting
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save the trained model
    model_filename = 'binary_classifier_model.joblib'
    dump(pipeline, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
    train_and_save_model(csv_path)
