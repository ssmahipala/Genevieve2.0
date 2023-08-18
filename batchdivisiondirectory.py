import os
import shutil
import math

def divide_images_into_batches(input_dir, batch_size):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    num_batches = math.ceil(len(image_files) / batch_size)

    for i in range(num_batches):
        batch_images = image_files[i * batch_size: (i + 1) * batch_size]
        batch_output_dir = os.path.join(input_dir, f'batch_{i + 1}')
        os.makedirs(batch_output_dir)

        for image_file in batch_images:
            source_path = os.path.join(input_dir, image_file)
            dest_path = os.path.join(batch_output_dir, image_file)
            shutil.copy(source_path, dest_path)

if __name__ == "__main__":
    input_directory = r'D:\My Work - Personal\Genevieve\Preprocessed\Sorted\512x512\batch_1'  # Replace with the path to your input directory
    batch_size = 100  # Number of images in each batch

    divide_images_into_batches(input_directory, batch_size)
    print(f"Images divided into batches of {batch_size} in {input_directory}")
