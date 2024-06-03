import pandas as pd

# Define the path to your CSV files
train_csv_path = 'csvData/train.csv'
test_csv_path = 'csvData/test.csv'
val_csv_path = 'csvData/val.csv'

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
val_df = pd.read_csv(val_csv_path)

# Count the number of images per emotion for each DataFrame
train_counts = train_df['emotion'].value_counts()
test_counts = test_df['emotion'].value_counts()
val_counts = val_df['emotion'].value_counts()

# Print the counts
print("Train set image counts per emotion:")
print(train_counts)
print("\nTest set image counts per emotion:")
print(test_counts)
print("\nValidation set image counts per emotion:")
print(val_counts)
