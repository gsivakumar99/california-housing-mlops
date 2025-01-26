import numpy as np

from src.data import load_data, preprocess_data
from src.model import HousePriceModel


def test_model_training():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Initialize and train model
    model = HousePriceModel()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Basic tests
    assert len(predictions) == len(y_test)
    assert isinstance(predictions, np.ndarray)
    assert all(isinstance(pred, np.float64) for pred in predictions)


def test_model_parameters():
    model = HousePriceModel(n_estimators=50, max_depth=5)
    params = model.get_params()

    assert params['n_estimators'] == 50
    assert params['max_depth'] == 5
