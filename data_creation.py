import yfinance as yf
import os
import pandas as pd

# Define the NASDAQ ticker symbol
nasdaq_ticker = "^IXIC"

# Define the start and end dates for the data
start_date = "2023-01-01"
end_date = "2024-12-31"

# Define the train/test split date
split_date = "2024-01-01"

# Download the data using yfinance
nasdaq_data = yf.download(nasdaq_ticker, start=start_date, end=end_date, interval="1h")

# Divide the data into training and testing sets
train_data = nasdaq_data[:split_date]
test_data = nasdaq_data[split_date:]

# Create the train and test directories if they don't exist
train_dir = "train"
test_dir = "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save the training and testing data to CSV files
train_data.to_csv(os.path.join(train_dir, "nasdaq_train.csv"))
test_data.to_csv(os.path.join(test_dir, "nasdaq_test.csv"))

# Print the first 5 rows of the training and testing data to verify that they have been saved correctly
print("Training data:")
print(train_data.head())
print("\nTesting data:")
print(test_data.head())
