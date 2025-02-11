import os
import tensorflow as tf
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.applications.resnet50 import preprocess_input
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Storage Config
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# Define class labels
CLASS_LABELS = ['Brown Rust', 'Healthy', 'Leaf Rust', 'Loose Smut', 'Septoria', 'Yellow Rust']

# Local model path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download files from Azure Blob Storage
def download_blob(blob_name, local_path):
    """Download a file from Azure Blob Storage if it does not exist locally."""
    if not os.path.exists(local_path):  # Avoid re-downloading
        print(f"Downloading {blob_name}...")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(local_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

# Download required model files if they are missing
model_files = {
    "saved_model.pb": "saved_model.pb",
    "variables.index": "variables/variables.index",
    "variables.data-00000-of-00001": "variables/variables.data-00000-of-00001"
}

for blob_name, local_path in model_files.items():
    full_local_path = os.path.join(MODEL_DIR, local_path)
    os.makedirs(os.path.dirname(full_local_path), exist_ok=True)  # Ensure directories exist
    download_blob(blob_name, full_local_path)

# Load TensorFlow model once (when the server starts)
print("Loading TensorFlow model...")
saved_model = tf.saved_model.load(MODEL_DIR)
predict_fn = saved_model.signatures['serving_default']
print("Model loaded successfully.")

@csrf_exempt
def predict(request):
    if request.method == "POST":
        # Get the uploaded image from the request
        image = request.FILES.get("image")
        if not image:
            return JsonResponse({"error": "No image provided"}, status=400)

        try:
            # Decode and preprocess the image
            image_data = tf.image.decode_image(image.read(), channels=3)
            image_data = tf.image.resize(image_data, [224, 224])
            image_data = tf.cast(image_data, tf.float32).numpy()
            image_data = np.expand_dims(image_data, axis=0)
            image_data = preprocess_input(image_data)

            # Predictions using the SavedModel
            predictions = predict_fn(tf.convert_to_tensor(image_data))
            output_tensor_name = list(predictions.keys())[0]
            predicted_values = predictions[output_tensor_name].numpy()
            predicted_class_index = np.argmax(predicted_values, axis=1)[0]
            predicted_class_name = CLASS_LABELS[predicted_class_index]
            predicted_probability = np.max(predicted_values, axis=1)[0] * 100

            return JsonResponse({
                "predicted_class": predicted_class_name,
                "probability": f"{predicted_probability:.2f}%"
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
