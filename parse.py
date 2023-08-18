import json
import os
import requests
from tqdm import tqdm

# Directory containing JSON files
json_directory = 'D:\My Work - Personal\Datasets\MidJourney_userPrompt_associated_with_their_images\Part_1'

# Create a directory to store images
if not os.path.exists('images'):
    os.makedirs('images')

# Lists to store user prompts and image URLs
user_prompts = []
image_urls = []

# Iterate through JSON files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            for message_list in data['messages']:
                for message in message_list:
                    if 'content' in message and 'attachments' in message:
                        user_prompt = message['content']
                        attachments = message['attachments']
                        for attachment in attachments:
                            if 'url' in attachment:
                                image_url = attachment['url']
                                user_prompts.append(user_prompt)
                                image_urls.append(image_url)

# Track the total number of images and initialize a tqdm progress bar
total_images = len(image_urls)
with tqdm(total=total_images, desc='Downloading Images', unit='image', dynamic_ncols=True) as pbar:
    # Download images
    for idx, url in enumerate(image_urls):
        response = requests.get(url)
        image_filename = f'images/image_{idx}.png'
        with open(image_filename, 'wb') as img_file:
            img_file.write(response.content)
        pbar.update(1)  # Update the progress bar

# Save user prompts to a text file
with open('user_prompts.txt', 'w') as txt_file:
    for prompt in user_prompts:
        txt_file.write(prompt + '\n')

# Display a message indicating completion
print(f'Images downloaded: {total_images}/{total_images}')
