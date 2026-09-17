import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    # Define file path (using raw string for Windows path)
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"

    try:
        # Load the dataset
        print("Loading dataset...")
        data = pd.read_csv(csv_path)

        # Verify target column exists
        if 'Class' not in data.columns:
            raise ValueError("Target column 'Class' not found in the dataset")

        # Separate features and target
        X = data.drop(columns=['Class'])
        y = data['Class']

        # Split data into train and validation sets with stratification
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=20260915,
            stratify=y
        )

        # Define preprocessing for numerical and categorical columns
        numerical_cols = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        categorical_cols = ['V2']

        # Create transformers with more robust imputation
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )

        # Create pipeline with more robust model parameters
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=20260915,
                n_jobs=-1
            ))
        ])

        # Train the model with progress bar
        print("Training model...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

        # Save the model to a file in current working directory
        model_path = "rf_classifier_pipeline.joblib"
        dump(model, model_path, compress=3)
        print(f"Model successfully saved to {model_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
