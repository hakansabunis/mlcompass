import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm

def train_and_save_model(csv_path, model_save_path='model.joblib'):
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Remove duplicate rows (based on mlcompass warning)
    df = df.drop_duplicates()
    print(f"Removed {len(df) - df.shape[0]} duplicate rows")

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            random_state=20260915,
            n_estimators=100,
            class_weight='balanced',
            max_depth=10
        ))
    ])

    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set (optional)
    test_score = pipeline.score(X_test, y_test)
    print(f"Model test accuracy: {test_score:.4f}")

    # Save the trained model
    dump(pipeline, model_save_path)
    print(f"Model saved to {model_save_path}")

    return pipeline

if __name__ == "__main__":
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"
    train_and_save_model(csv_path)
