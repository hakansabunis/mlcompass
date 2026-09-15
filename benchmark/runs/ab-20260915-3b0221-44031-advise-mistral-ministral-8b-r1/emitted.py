import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm import tqdm

def main():
    # Load the dataset
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r1\train.csv"
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Handle outliers using IQR method for features with known outliers
    features_with_outliers = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    for feature in features_with_outliers:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[feature] = np.where(X[feature] < lower_bound, lower_bound,
                             np.where(X[feature] > upper_bound, upper_bound, X[feature]))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize and train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    # Train model with progress bar
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {val_mse:.4f}")

    # Save the model and scaler
    dump(model, 'house_price_model.joblib')
    dump(scaler, 'scaler.joblib')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
