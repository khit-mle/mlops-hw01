import pickle

import pandas as pd
import pytest
from sklearn.metrics import r2_score


# Function to load dataset from a CSV file
def load_dataset_from_csv(filename):
    df = pd.read_csv(filename)
    X = df["X"].values.reshape(-1, 1)
    y = df["y"].values
    return X, y


# Function to load the trained model from the pickle file
def load_model(pickle_filename):
    with open(pickle_filename, "rb") as file:
        model = pickle.load(file)
    return model


# Test function to calculate the R² score for each dataset
@pytest.mark.parametrize(
    "dataset_filename",
    [
        "dataset_one_little_noise.csv",
        "dataset_two_little_noise.csv",
        "dataset_three_little_noise.csv",
        "dataset_noisy.csv",
    ],
)
def test_r2_score(dataset_filename):
    model = load_model("linear_regression_model.pkl")
    X, y = load_dataset_from_csv(dataset_filename)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    print(f"R² score for {dataset_filename}: {score}")
    assert score > 0.8, f"R² score is too low for {dataset_filename}: {score}"


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
