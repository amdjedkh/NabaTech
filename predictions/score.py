import json
import tensorflow as tf
import numpy as np
from azureml.core.model import Model

# Define class labels
CLASS_LABELS = ['Brown Rust', 'Healthy', 'Leaf Rust', 'Loose Smut', 'Septoria', 'Yellow Rust']

# Initialize the model
def init():
    global saved_model, predict_fn
    model_path = Model.get_model_path("wheat-disease-predictor")  # Retrieve the model path
    saved_model = tf.saved_model.load(model_path)  # Load the saved model
    predict_fn = saved_model.signatures['serving_default']  # Get the default serving signature

# Run predictions
def run(raw_data):
    try:
        # Parse the input data
        data = json.loads(raw_data)
        image_data = np.array(data["image"])  # Expecting a preprocessed numpy array
        
        if image_data.shape != (1, 224, 224, 3):
            return json.dumps({"error": "Invalid input shape. Expected shape is (1, 224, 224, 3)."})

        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)

        # Perform prediction
        predictions = predict_fn(image_tensor)  # Pass the preprocessed image through the model
        output_tensor_name = list(predictions.keys())[0]  # Get the output tensor's key
        predicted_values = predictions[output_tensor_name].numpy()
        predicted_class_index = np.argmax(predicted_values, axis=1)[0]  # Get the predicted class index
        predicted_class_name = CLASS_LABELS[predicted_class_index]  # Map index to class name
        predicted_probability = np.max(predicted_values, axis=1)[0] * 100  # Get the confidence score as a percentage

        # Return the prediction result
        return json.dumps({
            "predicted_class": predicted_class_name,
            "probability": f"{predicted_probability:.2f}%"
        })

    except Exception as e:
        return json.dumps({"error": str(e)})
