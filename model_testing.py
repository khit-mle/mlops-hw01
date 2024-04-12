import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


def load_preprocessed_data(file_path):
    data = np.load(file_path)
    return data["X"], data["y"]


def load_scaler(file_path):
    with open(file_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def inverse_transform(scaler, data, feature_index=0):
    # Create a dummy array with the same number of samples and the same number of features as during scaler fitting
    dummy = np.zeros((len(data), scaler.n_features_in_))
    # Place the data in the correct column corresponding to the feature index
    dummy[:, feature_index] = data.flatten()
    # Inverse transform the dummy array
    transformed = scaler.inverse_transform(dummy)
    # Extract the transformed data for the target feature
    return transformed[:, feature_index]


def evaluate_model(model, X_test, y_test, scaler, ticker):
    predictions = model.predict(X_test).flatten()  # Flatten predictions to a 1D array
    y_test = y_test.flatten()  # Ensure y_test is also a 1D array

    # Inverse transform predictions and actual values using the adjusted function
    predictions = inverse_transform(scaler, predictions, feature_index=0)
    y_test = inverse_transform(scaler, y_test, feature_index=0)

    mse = np.mean(np.square(predictions - y_test))
    rmse = np.sqrt(mse)
    print(f"MSE: {mse}, RMSE: {rmse}")

    # Plot predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label="Predictions")
    plt.plot(y_test, label="Actuals")
    plt.title(f"Stock Price Predictions vs Actuals for {ticker}")
    plt.legend()

    # Save the plot to a file in high quality
    plt.savefig(f"models/{ticker}_evaluation.png", format="png", dpi=300)
    plt.show()


# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Test model for each ticker
for ticker in ["abc", "xyz"]:
    # Load model, test data, and scaler
    model_path = f"models/{ticker}_lstm_model.h5"
    test_data_path = f"test/{ticker}_test.npz"
    scaler_path = f"train/{ticker}_scaler.pkl"

    model = load_model(model_path)
    X_test, y_test = load_preprocessed_data(test_data_path)
    scaler = load_scaler(scaler_path)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler, ticker)
