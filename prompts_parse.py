import os
import json

# Directory containing JSON files
json_directory = 'D:\My Work - Personal\Datasets\MidJourney_userPrompt_associated_with_their_images\Part_1'

# List to store user prompts
user_prompts = []

# Iterate through JSON files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_directory, filename)
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            for message_list in data['messages']:
                for message in message_list:
                    if 'content' in message:
                        user_prompt = message['content']
                        user_prompts.append(user_prompt)

# Save user prompts to a text file
output_file_path = 'user_prompts.txt'
with open(output_file_path, 'w', encoding='utf-8') as txt_file:
    for prompt in user_prompts:
        txt_file.write(prompt + '\n')

print(f'{len(user_prompts)} user prompts extracted and saved to {output_file_path}')
