from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('./plant_disease_detection_saved_model')

# Load categories
with open('./categories.json') as f:
    categories = json.load(f)

def preprocess_image(image):
    # Convert the image to a NumPy array
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # Adjust size as needed
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    image_array = preprocess_image(image)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map to category names
    predicted_label = categories.get(str(predicted_class), 'Unknown')

    return jsonify({'class': predicted_label, 'confidence': float(predictions[0][predicted_class])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
