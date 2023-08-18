from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Define input and output directories
input_dir = 'D:\My Work - Personal\Genevieve\images'  # Directory containing your original images
output_dir = 'D:\My Work - Personal\Genevieve\dataset\preprocessed_images'  # Directory to save preprocessed images

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize tqdm progress bar
pbar = tqdm(image_files, desc='Preprocessing Images', unit='image', dynamic_ncols=True)

# Create a log file for error files
error_log_file = open('error_log.txt', 'w')

# Loop through each image file
for image_file in pbar:
    try:
        # Load and resize the image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)
        target_size = (256, 256)  # Define the desired size
        image = image.resize(target_size, Image.LANCZOS)

        # Normalize pixel values to [0, 1] range
        image_array = np.array(image) / 255.0

        # Save the preprocessed image
        preprocessed_image = Image.fromarray((image_array * 255).astype(np.uint8))
        output_path = os.path.join(output_dir, image_file)
        preprocessed_image.save(output_path)

        # Update tqdm progress bar
        pbar.set_postfix(remaining=len(image_files) - pbar.n)  # Display remaining images count

    except Exception as e:
        # Log error and continue with the next image
        error_log_file.write(f"Error processing image: {image_file}\n")
        error_log_file.write(f"Error message: {str(e)}\n")
        error_log_file.write("-" * 50 + "\n")

# Close error log file
error_log_file.close()

# Close tqdm progress bar
pbar.close()

print("Preprocessing complete.")
