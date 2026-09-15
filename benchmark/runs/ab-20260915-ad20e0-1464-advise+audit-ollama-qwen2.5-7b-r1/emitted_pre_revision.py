import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
url = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
df = pd.read_csv(url)

# Drop duplicate rows to avoid inflated held-out scores
df.drop_duplicates(inplace=True)

# Split features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the fitted model to the current working directory
model_filename = 'trained_model.joblib'
dump(model, model_filename)

print(f"Model saved to {model_filename}")
