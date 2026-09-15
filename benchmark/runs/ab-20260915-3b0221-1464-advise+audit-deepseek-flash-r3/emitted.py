import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load the training data
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"
df = pd.read_csv(train_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Hold out a validation split (also seeded)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Define candidate models and their hyperparameter grids
models = {
    'logistic': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=1000))
        ]),
        {'clf__C': [0.1, 1, 10]}
    ),
    'random_forest': (
        RandomForestClassifier(random_state=SEED, class_weight='balanced'),
        {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }
    ),
    'gradient_boosting': (
        GradientBoostingClassifier(random_state=SEED),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    ),
    'hist_gradient_boosting': (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            'max_iter': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [None, 5, 10]
        }
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

best_score = -np.inf
best_model = None
best_name = None

# Evaluate each model using CV on the training split, then check on the validation split
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    val_pred = grid.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"{name}: cv={grid.best_score_:.4f}  val={val_acc:.4f}")
    if val_acc > best_score:
        best_score = val_acc
        best_model = grid.best_estimator_
        best_name = name

print(f"Best model: {best_name} with validation accuracy {best_score:.4f}")

# Refit the best model on the full dataset
best_model.fit(X, y)

# Save the fitted model to the current working directory
joblib.dump(best_model, 'model.joblib')
print("Model saved to model.joblib")
