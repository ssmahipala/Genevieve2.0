import os
import csv

def divide_csv_into_batches(input_file, output_dir, batch_size):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Read and store the header

        prompt_index = header.index("prompt")
        image_path_index = header.index("image_path")

        batch_number = 1
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                output_file = os.path.join(output_dir, f'batch_{batch_number}.csv')
                with open(output_file, 'w', newline='', encoding='utf-8') as batch_csv:
                    writer = csv.writer(batch_csv)
                    writer.writerow(header)
                    writer.writerows(batch)
                print(f'Saved batch {batch_number}: {output_file}')
                batch_number += 1
                batch = []

        # Write the last batch if it's not empty
        if batch:
            output_file = os.path.join(output_dir, f'batch_{batch_number}.csv')
            with open(output_file, 'w', newline='', encoding='utf-8') as batch_csv:
                writer = csv.writer(batch_csv)
                writer.writerow(header)
                writer.writerows(batch)
            print(f'Saved batch {batch_number}: {output_file}')

if __name__ == "__main__":
    input_csv = r'D:\My Work - Personal\Genevieve\datasets\New_datasets\512\datasettrain_dataset.csv'  # Replace with your input CSV file
    output_directory = r'D:\My Work - Personal\Genevieve\datasets\New_datasets\512'  # Replace with your desired output directory
    batch_size = 1000

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    divide_csv_into_batches(input_csv, output_directory, batch_size)
