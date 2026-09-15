import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm

# Set all random seeds for reproducibility
random.seed(20260915)
np.random.seed(20260915)

def train_and_save_model(csv_path, model_save_path='house_price_model.joblib'):
    """
    Train a model to predict house prices and save it to a file.
    Handles data cleaning and preprocessing.
    """
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Split data into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915
    )

    # Create a pipeline with preprocessing and model
    # Using RandomForest which doesn't require feature scaling
    # Removed scaler since RandomForest handles it internally
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=20260915,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        ))
    ])

    # Train the model with progress bar
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    test_score = pipeline.score(X_test, y_test)
    print(f"Model test R² score: {test_score:.4f}")

    # Save the model
    dump(pipeline, model_save_path)
    print(f"Model saved to {model_save_path}")

    return pipeline

if __name__ == "__main__":
    # Define the path to the CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"

    # Train and save the model
    trained_model = train_and_save_model(csv_path)
