import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

def main():
    # Define file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Identify numeric and categorical columns
        numeric_cols = ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        categorical_cols = ['V2']

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Create full pipeline with classifier
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            ))
        ])

        # Split data into train and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean CV F1 score: {np.mean(scores):.4f}")

        # Save the trained model to disk
        model_filename = 'trained_model.joblib'
        dump(model, model_filename, compress=3)
        print(f"Model saved as {model_filename}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
