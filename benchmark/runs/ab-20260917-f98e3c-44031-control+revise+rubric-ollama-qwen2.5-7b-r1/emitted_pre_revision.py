import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to file
dump(model, 'price_prediction_model.joblib')

print("Model training completed and saved.")
