import os
from datetime import datetime

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data import load_data, preprocess_data

# Create model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate and log metrics
        mlflow.log_metric("train_mse", mean_squared_error(y_train, y_train_pred))
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_test_pred))
        mlflow.log_metric("test_r2", r2_score(y_test, y_test_pred))

        # Infer the model signature
        signature = infer_signature(X_train, y_train_pred)

        # Create an input example
        input_example = X_train.iloc[:5]

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example
        )

        print(f"{model_name} logged to MLflow")


def train_models():
    mlflow.set_experiment("california_housing")
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Define models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5)),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=50, max_depth=10))
    ]

    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_models()
