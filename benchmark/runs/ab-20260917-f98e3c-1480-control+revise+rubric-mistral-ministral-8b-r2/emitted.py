import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from joblib import dump
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"
data = pd.read_csv(file_path)

# Remove duplicate rows (if any) before any processing
data = data.drop_duplicates()

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Check for class imbalance and use stratified sampling
class_counts = y.value_counts()
print("Class distribution:", class_counts)

# Split data into training and validation sets (stratified to preserve class distribution)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Define preprocessing for numerical and categorical columns
numerical_cols = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
categorical_cols = ['V2']

# Create transformers with imputation for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline with preprocessing and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        class_weight='balanced'  # Handle class imbalance
    ))
])

# Train the model only on training data
model.fit(X_train, y_train)

# Evaluate on validation data (data it was not trained on)
y_pred = model.predict(X_val)

# Use balanced accuracy which is robust to class imbalance
balanced_acc = balanced_accuracy_score(y_val, y_pred)
print(f"\nValidation Balanced Accuracy: {balanced_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save the trained model
model_path = 'trained_classifier.joblib'
dump(model, model_path)
print(f"\nModel saved to {model_path}")
