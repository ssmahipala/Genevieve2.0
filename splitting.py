import pandas as pd
from sklearn.model_selection import train_test_split

# Input directory path for the original dataset CSV
input_csv_path = r'D:\My Work - Personal\Genevieve\datasets\New_datasets\filtered1664.csv'

# Output directory path for the split datasets
output_directory = r'D:\My Work - Personal\Genevieve\dataset'

# Load the CSV file containing user prompts and image file paths
data = pd.read_csv(input_csv_path)

# Split the data into train, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the split datasets to CSV files in the specified output directory
train_data.to_csv(output_directory + 'train_dataset.csv', index=False)
validation_data.to_csv(output_directory + 'validation_dataset.csv', index=False)
test_data.to_csv(output_directory + 'test_dataset.csv', index=False)
