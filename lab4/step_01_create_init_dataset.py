import os

from catboost.datasets import titanic


# Create lab4 data dir if not exists
data_dir = "lab4/data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load train from Titanic dataset
train_df, _ = titanic()

# Select columns as per homework assignment
initial_df = train_df[["Pclass", "Sex", "Age"]]

# Save dataframe to csv which we would track with DVC
initial_df.to_csv(os.path.join(data_dir, "titanic.csv"), index=False, header=True)
