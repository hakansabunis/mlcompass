import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load the dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv'
data = pd.read_csv(file_path)

# Split into features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file in the current directory
dump(model, 'housing_price_model.joblib')
