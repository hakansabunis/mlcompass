import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv(r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r1\train.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
dump(model, 'trained_model.joblib')
