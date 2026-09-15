import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from joblib import dump
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline with imputation and scaling
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle potential missing values
    ('scaler', StandardScaler())  # Standardize features by removing the mean and scaling to unit variance
])

# Create a pipeline with preprocessing and model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        n_jobs=-1  # Use all available cores
    ))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Evaluation - Mean Squared Error: {mse:.6f}")

# Save the trained model to disk
model_file = "house_price_model.joblib"
dump(model, model_file)
print(f"Model saved to {model_file}")
