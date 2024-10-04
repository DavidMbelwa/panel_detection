
from flask import Flask, request, render_template, jsonify
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('solar_panel_detection_model.keras')

# Google Maps Static API Key (replace 'YOUR_GOOGLE_API_KEY' with your actual API key)
GOOGLE_MAPS_API_KEY = "AIzaSyC22VUhVRz1iaJ1F1nCH_5uDE5Kdlt4io0"

# Preprocessing function (as defined in the original code)
def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB').resize((101, 101))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Create additional features
    grayscale = np.mean(img_array, axis=-1, keepdims=True)
    dx, dy = np.gradient(grayscale[:, :, 0])
    edge_magnitude = np.sqrt(dx**2 + dy**2)
    texture = np.abs(grayscale - np.mean(grayscale))

    # Combine all features
    features = np.concatenate([img_array, grayscale, edge_magnitude[..., np.newaxis], texture], axis=-1)
    return np.expand_dims(features, axis=0)

# Route for the front-end
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to classify an image based on the location
@app.route('/classify', methods=['POST'])
def classify():
    location = request.form.get('location')
    size = request.form.get('size', '900x900')  # Default size is 400x400
    zoom = request.form.get('zoom', 19)  # Default zoom level is 18

    if not location:
        return jsonify({'error': 'No location provided'}), 400

    # Fetch the satellite image using Google Maps Static API with adjustable size and zoom
    static_map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?center={location}"
        f"&zoom={zoom}&size={size}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    )

    try:
        # Preprocess the image
        features = load_and_preprocess_image_from_url(static_map_url)

        # Predict if there are solar panels in the image
        prediction = model.predict(features)[0][0]
        prediction_label = "Solar panels detected" if prediction > 0.5 else "No solar panels detected"

        return jsonify({
            'location': location,
            'prediction': prediction_label,
            'image_url': static_map_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# API endpoint to handle feedback from the user
@app.route('/feedback', methods=['POST'])
def feedback():
    location = request.form.get('location')
    is_correct = request.form.get('is_correct') == 'true'  # Convert string to boolean

    # Log the feedback (for now, just print it; you could save it to a database or file)
    feedback_message = f"Feedback received for location '{location}': {'Correct' if is_correct else 'Incorrect'}"
    print(feedback_message)

    # Return a success message
    return jsonify({'message': 'Feedback submitted successfully'})


if __name__ == '__main__':
    app.run()
