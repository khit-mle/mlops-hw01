import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Constants
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2


def load_preprocessed_data(file_path):
    data = np.load(file_path)
    return data["X"], data["y"]


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(50),
        Dropout(DROPOUT_RATE),
        Dense(1),
    ])
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    return model


def train_model(model, X, y):
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    return model


# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and train model for each ticker
for ticker in ["abc", "xyz"]:
    train_file_path = f"train/{ticker}_train.npz"
    X_train, y_train = load_preprocessed_data(train_file_path)

    # Build and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
    lstm_model = build_lstm_model(input_shape)
    trained_model = train_model(lstm_model, X_train, y_train)

    # Save the trained model
    model_save_path = f"models/{ticker}_lstm_model.h5"
    os.makedirs("models", exist_ok=True)
    trained_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
