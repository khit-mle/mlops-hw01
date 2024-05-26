import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Get the current script's directory
script_dir = os.path.dirname(__file__)

# Read the CSV file from the data/full directory
df = pd.read_csv(os.path.join(script_dir, "data/full/full_dataset.csv"))

# Create new variables based on technical analysis indicators
# percentage change in close price
df["close_pct_change"] = df["close"].pct_change()

# 3-month moving average of volume
df["volume_ma"] = df["volume"].rolling(window=3).mean()

# 6-month moving average of retail sales
df["retail_sales_ma"] = df["retail_sales"].rolling(window=6).mean()

# 3-month moving average of commodity index
df["commodity_index_ma"] = df["commodity_index"].rolling(window=3).mean()

# percentage change in CPI
df["cpi_pct_change"] = df["cpi"].pct_change()

# 6-month moving average of unemployment rate
df["unemployment_rate_ma"] = df["unemployment_rate"].rolling(window=6).mean()

# Drop the first few rows since they don't have enough previous months' values
df = df.dropna()

# Sort the data by date in ascending order
df = df.sort_values(by="date")

# Shift the close column to create a lag feature for prediction
df["close_next_month"] = df["close"].shift(-1)

# Drop the last row since it won't have a corresponding close value for the next month
df = df.dropna()

# Define the preprocessing pipeline
numeric_features = [
    "close",
    "volume",
    "retail_sales",
    "commodity_index",
    "cpi",
    "unemployment_rate",
    "close_pct_change",
    "volume_ma",
    "retail_sales_ma",
    "commodity_index_ma",
    "cpi_pct_change",
    "unemployment_rate_ma",
]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])

# Split the data into training and testing sets using walk-forward optimization
train_size = int(0.8 * len(df))
train_df, test_df = df.head(train_size), df.tail(len(df) - train_size)

# Define the directories for train and test data inside the data directory
data_dir = os.path.join(script_dir, "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Create directories if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Save the unprocessed training and testing data to CSV files
train_df.to_csv(os.path.join(train_dir, "train_unprocessed.csv"), index=False)
test_df.to_csv(os.path.join(test_dir, "test_unprocessed.csv"), index=False)

# Preprocess the training and testing data
X_train = train_df.drop(["date", "close_next_month"], axis=1)
y_train = train_df["close_next_month"]
X_test = test_df.drop(["date", "close_next_month"], axis=1)
y_test = test_df["close_next_month"]

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Ensure the index is reset before combining
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Save the preprocessed training and testing data to CSV files
train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=numeric_features)
train_preprocessed_df["close_next_month"] = y_train
train_preprocessed_df["date"] = train_df["date"].reset_index(drop=True)
train_preprocessed_df.to_csv(os.path.join(train_dir, "train_preprocessed.csv"), index=False)

test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=numeric_features)
test_preprocessed_df["close_next_month"] = y_test
test_preprocessed_df["date"] = test_df["date"].reset_index(drop=True)
test_preprocessed_df.to_csv(os.path.join(test_dir, "test_preprocessed.csv"), index=False)
