import os
import csv

# Directory containing user prompts and images
prompts_directory = 'D:\My Work - Personal\Genevieve'
images_directory = 'D:\My Work - Personal\Genevieve\preprocessed'  # The directory where you downloaded images

# Create a directory to store the dataset
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Read user prompts from the text file
user_prompts = []
with open(os.path.join(prompts_directory, 'user_prompts.txt'), 'r', encoding='utf-8') as txt_file:
    user_prompts = txt_file.read().splitlines()

# Pair user prompts with image filenames
dataset = []
for idx, prompt in enumerate(user_prompts):
    image_filename = os.path.join(images_directory, f'image_{idx}.png')
    dataset.append((prompt, image_filename))

# Save the dataset to a CSV file
output_file_path = 'dataset.csv'
with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['prompt', 'image_filename'])  # Write header
    for prompt, image_filename in dataset:
        csv_writer.writerow([prompt, image_filename])

print(f'Dataset with {len(dataset)} samples created and saved to {output_file_path}')
