import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Define the path to the dataset
data_dir = "lab4/data"
file_path = os.path.join(data_dir, "titanic.csv")

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Initialize the OneHotEncoder
# drop='first' to avoid multicollinearity
encoder = OneHotEncoder(sparse_output=False, drop="first")

# Perform one-hot encoding on the 'Sex' column
encoded_sex = encoder.fit_transform(df[["Sex"]])

# Create a DataFrame with the one-hot encoded columns
encoded_sex_df = pd.DataFrame(encoded_sex, columns=encoder.get_feature_names_out(["Sex"]))

# Concatenate the original DataFrame and the one-hot encoded DataFrame
df = pd.concat([df.drop(columns=["Sex"]), encoded_sex_df], axis=1)

# Save the modified DataFrame back to the CSV file
df.to_csv(file_path, index=False, header=True)
