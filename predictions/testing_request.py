import json
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import requests
# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
    image = image.resize((224, 224))  # Resize to (224, 224)
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    image_array = preprocess_input(image_array)  # Normalize using ResNet50's preprocessing
    image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension
    return image_array

# Prepare request payload
def prepare_payload(image_array):
    return json.dumps({"image": image_array.tolist()})  # Convert to JSON-serializable format

# Example usage
image_path = r'C:\Users\dange\Downloads\yellooooo.webp'
image_array = preprocess_image(image_path)
payload = prepare_payload(image_array)

# Send to scoring endpoint
url = "https://mymodel-endpoint.spaincentral.inference.ml.azure.com/score"
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=payload, headers=headers)

print("Response:", response.json())
