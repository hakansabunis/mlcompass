import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import os

def load_and_prepare_data(file_path):
    """Load the CSV file and prepare the data for modeling."""
    # Load the data
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    return X, y, categorical_cols, numerical_cols

def create_preprocessor(categorical_cols, numerical_cols):
    """Create a column transformer for preprocessing."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor

def train_and_save_model(file_path, output_model_path='trained_model.pkl'):
    """Train a model and save it to a file."""
    # Load and prepare data
    X, y, categorical_cols, numerical_cols = load_and_prepare_data(file_path)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(categorical_cols, numerical_cols)

    # Create full pipeline with classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_path = os.path.join(os.getcwd(), output_model_path)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"

    # Train and save the model
    train_and_save_model(csv_path)
