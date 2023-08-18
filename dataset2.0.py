import os
import pandas as pd
from tqdm import tqdm

# Paths
source_directory = r'D:\My Work - Personal\Genevieve\Preprocessed\Sorted\2048x2048'
old_dataset_path = r'D:\My Work - Personal\Genevieve\dataset.csv'
new_dataset_path = r'D:\My Work - Personal\Genevieve\dataset\filtered.csv'

# Read the old dataset CSV
old_dataset = pd.read_csv(old_dataset_path)

# Create a list to store rows for the new dataset
new_dataset_rows = []

# Get the total number of files in the source directory
total_files = len(os.listdir(source_directory))

# Iterate through files in the source directory with a progress bar
for source_image_file_name in tqdm(os.listdir(source_directory), total=total_files, desc="Creating new dataset"):
    # Extract the image file name from the path
    image_file_name = os.path.basename(source_image_file_name)

    # Find the corresponding row in the old dataset
    matching_row = old_dataset[old_dataset['image_filename'].str.endswith(image_file_name)]

    if not matching_row.empty:
        # Extract the prompt from the matching row
        prompt = matching_row['prompt'].values[0]

        # Append a new row to the new dataset
        new_dataset_rows.append({'prompt': prompt, 'image_filename': source_image_file_name})

# Create a new DataFrame for the new dataset
new_dataset = pd.DataFrame(new_dataset_rows)

# Save the new dataset to a CSV file
new_dataset.to_csv(new_dataset_path, index=False)

print(f"New dataset with {len(new_dataset)} samples created and saved to {new_dataset_path}")
