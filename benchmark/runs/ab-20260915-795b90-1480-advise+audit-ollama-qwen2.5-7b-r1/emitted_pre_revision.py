import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed0-r1\train.csv"
df = pd.read_csv(file_path)

# Drop duplicate rows to avoid inflation in held-out scores
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model to a file in the current working directory
model_filename = 'trained_model.joblib'
dump(model, model_filename)

print(f"Model saved to {model_filename}")
