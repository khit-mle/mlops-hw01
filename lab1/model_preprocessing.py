import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SEQUENCE_LENGTH = 10  # Number of days in each sequence


def load_data(file_path):
    return pd.read_csv(file_path)


def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled, scaler


def create_sequences(data, sequence_length):
    sequences = []
    target = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i : i + sequence_length].values)
        target.append(data.iloc[i + sequence_length].values[0])  # Assuming the first column is the target
    return np.array(sequences), np.array(target)


def preprocess_data(file_path):
    # Load data
    df = load_data(file_path)

    # Drop the date column for processing
    df.drop("date", axis=1, inplace=True)

    # Normalize data
    df_normalized, scaler = normalize_data(df)

    # Create sequences
    X, y = create_sequences(df_normalized, SEQUENCE_LENGTH)

    return X, y, scaler


def save_preprocessed_data(X, y, file_path):
    np.savez(file_path, X=X, y=y)


# Ensure the directories exist
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Preprocess and save training and testing data for both stocks
for ticker in ["abc", "xyz"]:
    train_X, train_y, train_scaler = preprocess_data(f"train/{ticker}_train.csv")
    test_X, test_y, test_scaler = preprocess_data(f"test/{ticker}_test.csv")

    # Save preprocessed data
    save_preprocessed_data(train_X, train_y, f"train/{ticker}_train.npz")
    save_preprocessed_data(test_X, test_y, f"test/{ticker}_test.npz")

    # Optionally, save scaler for inverse transformation during prediction
    with open(f"train/{ticker}_scaler.pkl", "wb") as f:
        pickle.dump(train_scaler, f)
