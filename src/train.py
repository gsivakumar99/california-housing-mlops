import os
from typing import Tuple

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.data import load_data, preprocess_data

# Create model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
LATEST_DIR = os.path.join(MODEL_DIR, "latest")
os.makedirs(LATEST_DIR, exist_ok=True)


def train_and_log_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Train model, calculate metrics, and log to MLflow.

    Args:
        model: Scikit-learn model
        model_name: Name of the model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
    """
    # Train the model
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"{model_name} Results:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # Save Random Forest model as model.pkl
    if isinstance(model, RandomForestRegressor):
        model_path = os.path.join(LATEST_DIR, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)

        # Infer the model signature
        signature = infer_signature(X_train, y_train_pred)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=X_train.iloc[:5],
        )


def get_models() -> list[Tuple[str, BaseEstimator]]:
    """Return list of model names and their instances."""
    return [
        ("Linear_Regression", LinearRegression()),
        ("Decision_Tree", DecisionTreeRegressor(max_depth=5)),
        (
            "Random_Forest",
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
        ),
    ]


def train_models() -> None:
    """Train and evaluate all models."""
    # Set MLflow experiment
    mlflow.set_experiment("california_housing")

    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessing completed.")

    # Train and evaluate each model
    models = get_models()
    for model_name, model in models:
        print(f"\nTraining {model_name}...")
        train_and_log_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )


if __name__ == "__main__":
    train_models()
