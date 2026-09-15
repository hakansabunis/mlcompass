import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import os

# Set random seeds for reproducibility
random.seed(20260915)
np.random.seed(20260915)

def train_and_save_model(csv_path, model_save_path='trained_model.joblib'):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Remove duplicate rows to avoid data leakage
    df = df.drop_duplicates()
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Define numeric and categorical features
    numeric_features = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    categorical_features = ['V2']

    # Create transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=20260915))
    ])

    # Split data into train and test sets (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {os.path.abspath(model_save_path)}")

    return model_save_path

if __name__ == "__main__":
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"
    train_and_save_model(csv_path)
