from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import pickle
from PIL import Image
import requests
from io import BytesIO



# Google Maps API key
API_KEY = "AIzaSyC22VUhVRz1iaJ1F1nCH_5uDE5Kdlt4io0"


app = Flask(__name__)


# Set up paths
base_dir = './' # Modify this if needed
keras_file_path = "solar_panel_detection_model.keras"
pickle_file_path = "solar_panel_detection_model.pkl"


# Load the Keras model
model = tf.keras.models.load_model(keras_file_path)
print("Keras model loaded successfully!")


# The fuction to get the coordinates of the location
def get_coordinates(location_name):
    """Get the latitude and longitude of a location using Google Geocoding API."""
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={API_KEY}"
    response = requests.get(geocode_url)
    data = response.json()

    if data['status'] == 'OK':
        # Extract latitude and longitude from the response
        location = data['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        return lat, lng
    else:
        raise Exception("Error: Unable to fetch coordinates.")


# The function to fetch the satelite imagery
def get_satellite_image(lat, lng, zoom=15, size="400x400"):
    """Get a satellite image of a location using Google Maps Static API."""
    map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"
    response = requests.get(map_url)

    if response.status_code == 200:
        # Convert the response content to an image
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise Exception("Error: Unable to fetch satellite image.")


# The function to process the image
def load_and_preprocess_image(img):
    img = img.resize((101, 101))  # Resize the image
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize the image array

    # Create the grayscale image by taking the mean along the color channel axis
    grayscale = np.mean(img_array, axis=-1)  # Result is a 2D array (height, width)

    # Compute the gradients for the 2D grayscale image
    dx, dy = np.gradient(grayscale)  # This will work since grayscale is now 2D

    # Additional features
    edge_magnitude = np.sqrt(dx**2 + dy**2)  # Compute edge magnitude
    texture = np.abs(grayscale - np.mean(grayscale))  # Compute texture

    # Combine all features into one array
    features = np.concatenate(
        [
            img_array,  # Original image array (3D)
            grayscale[..., np.newaxis],  # Add grayscale back with a new axis
            edge_magnitude[..., np.newaxis],  # Add edge magnitude with a new axis
            texture[..., np.newaxis]  # Add texture with a new axis
        ],
        axis=-1  # Concatenate along the last axis (channels)
    )

    # Add batch dimension (required for model prediction)
    return np.expand_dims(features, axis=0)


# The function to classify the image
def classify_image(image):
    
     # Preprocess the image
    processed_image = load_and_preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0

    return "There are solar panels" if predicted_class == 1 else "There are no solar panels in your area."


# Define the route for prediction
@app.route('/', methods=['GET','POST'])

def index():
    if request.method == "POST":
        # User location
        location_name = request.form.get("location")

        # Fetching coordinates for the location
        coordinates = get_coordinates(location_name)

        if not coordinates:
            return "Error: Could not find the location."
        
        lat, lng = coordinates

        # Fetching the satellite image
        image = get_satellite_image(lat, lng)

        if not image:
            return "Error: Could not fetch satellite image"
        
        classification_result = classify_image(image)

        # Save the image to send it to the user interface
        img_io = BytesIO()
        image.save(img_io, "PNG")
        img_io.seek(0)

        # Send the image and classification result to be rendered in the HTML
        return render_template("index.html", image_url=request.url, classification=classification_result)

    # For GET request, simply render the form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
