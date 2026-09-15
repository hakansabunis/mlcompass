import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv'
df = pd.read_csv(file_path)

# Drop duplicate rows to avoid inflating held-out scores
df.drop_duplicates(inplace=True)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Score the held-out set
score = model.score(X_test, y_test)
print(f"Model accuracy on held-out test set: {score}")

# Save the model to the current working directory
dump(model, 'trained_model.joblib')

print("Model trained and saved successfully.")
