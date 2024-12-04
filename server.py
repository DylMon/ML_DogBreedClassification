import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "dog_breed_classifier.h5"
model = load_model(MODEL_PATH)

# List of breeds (update this with your actual class names)
breeds = ["Chihuahua", "Golden Retriever", "Beagle"]  # Replace with actual breeds

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define a prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the uploaded file
    file = request.files["file"]

    try:
        # Open and preprocess the image
        img = Image.open(file)
        img_array = preprocess_image(img)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]

        # Get the breed name
        predicted_breed = breeds[predicted_index]
        return jsonify({
            "breed": predicted_breed,
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
