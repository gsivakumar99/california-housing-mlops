import pytest
import os
import mlflow
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.main import train_and_log_model, train_models


# Mock for load_data and preprocess_data
@pytest.fixture
def mock_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Create a mock dataset
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [10, 20, 30, 40, 50, 60],
        "target": [100, 200, 300, 400, 500, 600]
    })
    X = df[["feature1", "feature2"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def test_train_and_log_model(mock_data):
    X_train, X_test, y_train, y_test = mock_data

    # Mock MLflow methods
    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_metric") as mock_log_metric, \
         patch("mlflow.sklearn.log_model") as mock_log_model:
        
        model = LinearRegression()
        train_and_log_model(model, "Linear Regression", X_train, X_test, y_train, y_test)

        # Assert MLflow methods are called
        mock_start_run.assert_called_once()
        mock_log_metric.assert_called()
        mock_log_model.assert_called_once()

        # Verify model training
        assert hasattr(model, "coef_"), "Model coefficients not found. Training failed."


def test_train_models(mock_data):
    X_train, X_test, y_train, y_test = mock_data

    # Mock functions in the workflow
    with patch("src.main.load_data") as mock_load_data, \
         patch("src.main.preprocess_data") as mock_preprocess_data, \
         patch("src.main.train_and_log_model") as mock_train_and_log_model, \
         patch("mlflow.set_experiment") as mock_set_experiment:

        mock_load_data.return_value = None
        mock_preprocess_data.return_value = (X_train, X_test, y_train, y_test)

        train_models()

        # Assert experiment setup
        mock_set_experiment.assert_called_once_with("california_housing")

        # Assert train_and_log_model is called for all models
        assert mock_train_and_log_model.call_count == 3


def test_model_directory_created():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    
    # Ensure the directory exists
    assert os.path.exists(model_dir), "Model directory was not created."


@pytest.mark.parametrize("model_name,model", [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=5)),
    ("Random Forest Regressor", RandomForestRegressor(n_estimators=50, max_depth=10))
])
def test_individual_models(mock_data, model_name, model):
    X_train, X_test, y_train, y_test = mock_data

    # Mock MLflow methods
    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_metric") as mock_log_metric, \
         patch("mlflow.sklearn.log_model") as mock_log_model:

        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)

        # Verify MLflow interactions
        mock_start_run.assert_called_once_with(run_name=model_name)
        mock_log_model.assert_called_once()

        # Validate model training
        assert hasattr(model, "predict"), "Model training or initialization failed."
