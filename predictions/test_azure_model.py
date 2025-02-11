import base64
import json
import requests

# Path to the image
image_path = "C:\\Users\\dange\\Downloads\\smut.jpg"

# Encode image to base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create the payload
payload = json.dumps({"image": base64_image})

# Set headers and endpoint
endpoint = "http://44404246-c28e-4e07-b9c1-2a1b7d7afe4f.spaincentral.azurecontainer.io/score"
api_key = "VfqZm5hpZn9oSW2Bvwc4ZzqKmg4AkhHE"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Send the request
response = requests.post(endpoint, data=payload, headers=headers)

# Print the result
print("Prediction Result:", response.text)
