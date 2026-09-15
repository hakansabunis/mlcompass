import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(20260915)

# Load the dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv'
data = pd.read_csv(file_path)

# Remove duplicate rows (as advised by mlcompass)
data = data.drop_duplicates()

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Define preprocessing for numeric and categorical columns
numeric_features = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
categorical_features = ['V2']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=20260915, n_estimators=100))
])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=20260915, stratify=y
)

# Train the model with progress bar
print("Training model...")
with tqdm(total=100) as pbar:
    model.fit(X_train, y_train, joblib_verbose=0)
    # Update progress bar (RandomForest doesn't provide direct progress, so we'll simulate)
    for _ in range(100):
        pbar.update(1)

# Evaluate on validation set
y_pred = model.predict(X_val)
print("\nValidation Set Performance:")
print(classification_report(y_val, y_pred))

# Save the trained model
model_filename = 'trained_model.joblib'
dump(model, model_filename)
print(f"\nModel saved as {model_filename}")
