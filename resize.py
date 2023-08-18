from PIL import Image, ImageOps
import os
from tqdm import tqdm

def resize_to_square(input_path, output_path, target_size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = os.listdir(input_path)
    for filename in tqdm(file_list, desc="Resizing images"):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)

        if os.path.isfile(input_file):
            img = Image.open(input_file)

            width, height = img.size
            if width != height:
                new_size = (target_size, target_size)
                resized_img = ImageOps.fit(img, new_size, method=Image.LANCZOS)
                resized_img.save(output_file, quality=95)
            else:
                img.save(output_file, quality=95)

if __name__ == "__main__":
    input_directory = "D:\My Work - Personal\Genevieve\images"
    output_directory = "D:\My Work - Personal\Genevieve\Preprocessed"
    target_size = 1024

    resize_to_square(input_directory, output_directory, target_size)
    print("Image preprocessing completed.")
