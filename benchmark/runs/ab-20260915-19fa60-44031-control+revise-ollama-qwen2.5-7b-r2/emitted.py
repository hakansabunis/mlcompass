import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load the dataset
df = pd.read_csv(r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed1-r2\train.csv')

# Separate features and target
X = df.drop(columns=['price'])
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to file in the current working directory
dump(model, 'price_prediction_model.joblib')
print("Model trained and saved as 'price_prediction_model.joblib'")
