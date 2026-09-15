import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]

def train_and_save_model():
    # Define file path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"

    # Load the dataset
    try:
        data = pd.read_csv(csv_path)
        print("Dataset loaded successfully with shape:", data.shape)

        # Check for missing values
        print("\nMissing values per column:")
        print(data.isnull().sum())

        # Verify target column exists
        if 'price' not in data.columns:
            raise ValueError("Target column 'price' not found in the dataset")

        # Separate features and target
        features = data.columns.drop('price')
        X = data[features]
        y = data['price']

        # Split data into train and test sets with stratification if possible
        # For regression, we'll use simple random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=20260915
        )

        # Create preprocessing pipeline
        preprocessor = Pipeline([
            ('selector', FeatureSelector(features)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                random_state=20260915,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                n_jobs=-1
            ))
        ])

        # Train the model
        print("\nTraining model...")
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\nModel evaluation:")
        print(f"Test MSE: {mse:.2f}")
        print(f"Test R² Score: {r2:.4f}")

        # Save the trained model
        model_filename = "house_price_model.joblib"
        joblib.dump(model, model_filename)
        print(f"\nModel saved successfully to {os.path.abspath(model_filename)}")

        # Print feature importances
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
            features = features.tolist()
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print("\nTop 5 most important features:")
            print(feature_importance.head(5))

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    train_and_save_model()
