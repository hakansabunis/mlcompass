import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
df = pd.read_csv(file_path)

# Drop duplicate rows
df.drop_duplicates(inplace=True)
assert df.shape[0] == 524 - 125, "Duplicate rows were not properly removed"

# Split features and target
X = df[['V1', 'V2', 'V3', 'V4']]
y = df['Class']

# Handle class imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to the current working directory
model_filename = 'trained_model.joblib'
dump(model, model_filename)
print(f"Model saved to {model_filename}")
