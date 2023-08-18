from PIL import Image, ImageOps
import os
from tqdm import tqdm


def resize_to_square(input_path, output_path, target_size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processed_images = set(os.listdir(output_path))

    file_list = os.listdir(input_path)
    skipped_images = []

    for filename in tqdm(file_list, desc="Resizing images"):
        if filename in processed_images:
            continue

        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)

        if os.path.isfile(input_file):
            try:
                img = Image.open(input_file)
                width, height = img.size

                if width != height:
                    new_size = (target_size, target_size)
                    resized_img = ImageOps.fit(img, new_size, method=Image.LANCZOS)
                    resized_img.save(output_file, quality=95)
                else:
                    img.save(output_file, quality=95)

                processed_images.add(filename)
            except Exception as e:
                skipped_images.append(filename)
                print(f"Error processing {filename}: {str(e)}")

    if skipped_images:
        with open('skipped_images_log.txt', 'w') as log_file:
            log_file.write("Skipped images:\n")
            for image_name in skipped_images:
                log_file.write(image_name + '\n')


if __name__ == "__main__":
    input_directory = "D:\My Work - Personal\Genevieve\images"
    output_directory = "D:\My Work - Personal\Genevieve\Preprocessed"
    target_size = 1024

    resize_to_square(input_directory, output_directory, target_size)
    print("Image preprocessing completed.")
