from PIL import Image
import os
from tqdm import tqdm
import json
import time
import shutil


def sort_images_by_resolution(input_path, output_root, max_retries=10, retry_delay=0.1):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    processed_images = set()
    checkpoint_file = os.path.join(output_root, 'checkpoint.json')

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as checkpoint:
            processed_images = set(json.load(checkpoint))

    file_list = os.listdir(input_path)
    total_images = len(file_list)
    remaining_images = total_images - len(processed_images)

    images_to_delete = []

    for filename in tqdm(file_list, desc="Sorting images", total=total_images, initial=total_images - remaining_images):
        if filename in processed_images:
            continue

        input_file = os.path.join(input_path, filename)

        if os.path.isfile(input_file):
            retries = 0
            success = False

            while retries < max_retries:
                try:
                    img = Image.open(input_file)
                    width, height = img.size

                    resolution_dir = os.path.join(output_root, f"{width}x{height}")
                    if not os.path.exists(resolution_dir):
                        os.makedirs(resolution_dir)

                    output_file = os.path.join(resolution_dir, filename)
                    shutil.copy2(input_file, output_file)  # Copy the file

                    images_to_delete.append(input_file)  # Mark for deletion

                    processed_images.add(filename)
                    remaining_images -= 1

                    # Update checkpoint
                    with open(checkpoint_file, 'w') as checkpoint:
                        json.dump(list(processed_images), checkpoint)

                    success = True
                    break

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    retries += 1
                    time.sleep(retry_delay)

            if not success:
                print(f"Failed to process {filename} after {max_retries} retries.")

    # Delete the original files
    for file_to_delete in images_to_delete:
        try:
            os.remove(file_to_delete)
        except Exception as e:
            print(f"Error deleting {file_to_delete}: {str(e)}")

    # Remove the checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("Image sorting completed.")

if __name__ == "__main__":
    input_directory = "D:\My Work - Personal\Genevieve\Preprocessed"
    output_directory = "D:\My Work - Personal\Genevieve\Preprocessed\Sorted"

    sort_images_by_resolution(input_directory, output_directory)


