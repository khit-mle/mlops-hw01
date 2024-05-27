import os
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Get the current script's directory
script_dir = os.path.dirname(__file__)

# Define the directories for the preprocessed test data and saved model
test_dir = os.path.join(script_dir, "data", "test")
model_dir = os.path.join(script_dir, "models")

# Load the preprocessed testing data
test_preprocessed_df = pd.read_csv(os.path.join(test_dir, "test_preprocessed.csv"))

# Separate features and target variable
X_test = test_preprocessed_df.drop(["date", "close_next_month"], axis=1)
y_test = test_preprocessed_df["close_next_month"]

# Load the trained model from the pickle file
model_file = os.path.join(model_dir, "linear_regression_model.pickle")
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Make predictions
y_test_pred = model.predict(X_test)

# Evaluate the model
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")
print(f"Test R2: {test_r2}")
