import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

# Split into train, validation, and test sets
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.25, random_state=42)

# Save datasets
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
