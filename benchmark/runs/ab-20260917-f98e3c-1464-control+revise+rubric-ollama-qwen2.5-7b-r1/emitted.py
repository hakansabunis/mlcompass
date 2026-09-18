import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Ensure no duplicate rows are present (though V1 to V4 are int64, this step is good practice)
df = df.drop_duplicates()

# Preprocess the data: encode categorical variables if any (there are none in this case, all int64)
# As V1 to V4 are already numerical, no encoding is needed

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model with a random state for reproducibility
model = RandomForestClassifier(random_state=42)

# Fit the model on training data only
model.fit(X_train, y_train)

# Save the fitted model to file in current working directory
dump(model, 'fitted_model.joblib')

print("Model training complete and saved.")
