import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_name=r"./data/california_housing.csv"):
    # Load California Housing dataset from CSV
    df = pd.read_csv(file_name)
    return df


def preprocess_data(df):
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
