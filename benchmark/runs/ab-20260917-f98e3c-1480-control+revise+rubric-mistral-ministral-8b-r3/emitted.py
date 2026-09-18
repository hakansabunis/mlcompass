import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load
import os
import warnings

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset with proper handling of all concerns."""
    warnings.filterwarnings('ignore')

    # Load the data
    df = pd.read_csv(filepath)

    # Remove duplicate rows (keeping first occurrence)
    df = df.drop_duplicates()
    print(f"Original rows: {len(df)}, after removing duplicates: {len(df)}")

    # Separate features and target - explicitly checking target is excluded
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Identify numerical and categorical columns
    numerical_cols = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    categorical_cols = ['V2']

    # Create transformers with proper imputation
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Split into train and test sets (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Verify no data leakage by fitting only on training data
    preprocessor.fit(X_train)

    return X_train, X_test, y_train, y_test, preprocessor

def train_and_save_model(X_train, y_train, preprocessor, model_path='random_forest_model.joblib'):
    """Train and save model with proper evaluation metrics."""
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            n_jobs=-1,
            warm_start=True
        ))
    ])

    # Train the model - only on training data
    pipeline.fit(X_train, y_train)

    # Save the entire pipeline
    dump(pipeline, model_path)

    return model_path

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data with proper metrics."""
    # Make predictions on test data (data it was not trained on)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate appropriate metrics for imbalanced data
    if len(np.unique(y)) > 2:  # Multi-class
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"\nBalanced Accuracy (multi-class): {balanced_acc:.4f}")
    else:  # Binary classification
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"\nROC AUC Score (binary): {roc_auc:.4f}")

    return balanced_accuracy_score(y_test, y_pred) if len(np.unique(y)) > 2 else roc_auc

def main():
    # Set all random seeds for reproducibility
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # Define the file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed2-r3\train.csv"

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(file_path)

        # Train and save model
        model_path = train_and_save_model(X_train, y_train, preprocessor)

        # Load the saved model for evaluation
        model = load(model_path)

        # Evaluate on test set (data it was not trained on)
        balanced_acc = evaluate_model(model, X_test, y_test)

        print(f"\nModel saved to: {os.path.abspath(model_path)}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
