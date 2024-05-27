import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Get the current script's directory
script_dir = os.path.dirname(__file__)

# Define the directories for the preprocessed train and test data
train_dir = os.path.join(script_dir, "data", "train")
test_dir = os.path.join(script_dir, "data", "test")

# Load the preprocessed training and testing data
train_preprocessed_df = pd.read_csv(os.path.join(train_dir, "train_preprocessed.csv"))
test_preprocessed_df = pd.read_csv(os.path.join(test_dir, "test_preprocessed.csv"))

# Separate features and target variable
X_train = train_preprocessed_df.drop(["date", "close_next_month"], axis=1)
y_train = train_preprocessed_df["close_next_month"]
X_test = test_preprocessed_df.drop(["date", "close_next_month"], axis=1)
y_test = test_preprocessed_df["close_next_month"]

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")
print(f"Train R2: {train_r2}")
print(f"Test R2: {test_r2}")

# Define the directory for saving the model
model_dir = os.path.join(script_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model as a pickle file
model_file = os.path.join(model_dir, "linear_regression_model.pickle")
with open(model_file, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_file}")
