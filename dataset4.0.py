import os
import pandas as pd
from tqdm import tqdm

# Paths
dataset_path = r'D:\My Work - Personal\Genevieve\dataset\filtered.csv'
image_directory = r'D:\My Work - Personal\Genevieve\Preprocessed\Sorted\2048x2048'

# Read the existing dataset
dataset = pd.read_csv(dataset_path)

# Create a list to store rows for the updated dataset
updated_dataset_rows = []

# Get the total number of rows in the dataset
total_rows = len(dataset)

# Iterate through rows in the dataset with a progress bar
for _, row in tqdm(dataset.iterrows(), total=total_rows, desc="Updating dataset"):
    # Extract the file name from the image_filename column
    file_name = os.path.basename(row['image_filename'])

    # Construct the full path to the image file
    image_path = os.path.join(image_directory, file_name)

    # Add the path to the row
    row['image_path'] = image_path

    # Append the updated row to the list
    updated_dataset_rows.append(row)

# Create a new DataFrame for the updated dataset
updated_dataset = pd.DataFrame(updated_dataset_rows)

# Save the updated dataset to a CSV file
updated_dataset.to_csv(dataset_path, index=False)

print(f"Dataset with paths updated and saved to {dataset_path}")
