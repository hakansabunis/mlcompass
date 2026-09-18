import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    # Load the data
    df = pd.read_csv(file_path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor

def train_and_evaluate_model(X, y, preprocessor):
    """Train and evaluate a RandomForest classifier with proper handling of imbalance"""
    # Check for class imbalance and handle if needed
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.max() / class_counts.min() > 2:
        print("Class imbalance detected. Using balanced accuracy score.")
        # Stratified split handles imbalance by preserving class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Create pipeline with preprocessing and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=10
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, X_test, y_test

def save_model(model, output_path='random_forest_model.joblib'):
    """Save the trained model to a file"""
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")
    return output_path

def main():
    # Define the file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data(file_path)

    # Train and evaluate model
    model, X_test, y_test = train_and_evaluate_model(X, y, preprocessor)

    # Save the model
    save_model(model)

if __name__ == "__main__":
    main()
