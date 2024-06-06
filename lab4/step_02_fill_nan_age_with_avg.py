import os

import pandas as pd


# Define the path to the dataset
data_dir = "lab4/data"
file_path = os.path.join(data_dir, "titanic.csv")

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Fill NaN values in the 'Age' column with the mean age
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Save the modified DataFrame back to the CSV file
df.to_csv(file_path, index=False, header=True)
