import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    # Load California Housing dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df


def preprocess_data(df):
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return train_test_split(X, y, test_size=0.2, random_state=42)
