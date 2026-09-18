import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("C:\\Users\\SABUNIS\\AppData\\Local\\Temp\\mlcab-1464-seed2-r3\\train.csv")

# Separate features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)

# Save the fitted model to a file in the current working directory
joblib_file = "trained_model.joblib"
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")
