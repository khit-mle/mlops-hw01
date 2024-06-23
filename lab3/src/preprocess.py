import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    X = data.drop("species", axis=1)
    y = data["species"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    return preprocess_data(data)
