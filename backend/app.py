from flask import Flask, request, jsonify
import torch
from torchvision.utils import save_image
from stylegan2.models import Generator

app = Flask(__name__)

# Load the trained StyleGAN2 model checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/stylegan2_checkpoint.pth'  # Adjust the path to your trained checkpoint
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()


@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        # Parse request data (you may need to adjust the request format)
        request_data = request.get_json()
        # Example: request_data = {'style': 'portrait', 'size': 256}

        # Process the request and generate an image using StyleGAN2
        with torch.no_grad():
            # Generate a random noise tensor
            noise = torch.randn(1, 512, 1, 1, device=device)

            # Forward pass through the generator
            generated_image = generator(noise)

        # Save the generated image
        output_path = 'generated_images/generated_image.png'  # Adjust the path as needed
        save_image(generated_image, output_path)

        # Return the path to the generated image
        return jsonify({'image_path': output_path})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
