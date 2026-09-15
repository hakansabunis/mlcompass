import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load the dataset
df = pd.read_csv(r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv')

# Separate features and target
X = df.drop(columns=['price'])
y = df['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model to a file
dump(model, 'trained_model.joblib')
