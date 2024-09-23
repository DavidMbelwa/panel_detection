from flask import Flask, request, render_template, send_file
import requests
from PIL import Image
from io import BytesIO
import ee  # Import the Earth Engine API
import os

# Initialize the Earth Engine API
ee.Initialize()

# Flask app initialization
app = Flask(__name__)

# Google Maps API key (optional if using Google Maps for fallback)
API_KEY = "AIzaSyC22VUhVRz1iaJ1F1nCH_5uDE5Kdlt4io0"

# The function to get the coordinates of the location
def get_coordinates(location_name):
    """Get the latitude and longitude of a location using Google Geocoding API."""
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={API_KEY}"
    response = requests.get(geocode_url)
    data = response.json()

    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        return lat, lng
    else:
        raise Exception("Error: Unable to fetch coordinates.")

# The function to get a satellite image from Google Earth Engine
def get_google_earth_image(lat, lng):
    """Fetch high-quality satellite imagery using Google Earth Engine."""
    
    # Define the location point using latitude and longitude
    point = ee.Geometry.Point([lng, lat])

    # Get Sentinel-2 satellite image for the point
    image = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first()

    # Define visualization parameters for true-color imagery (RGB bands)
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue bands for true color
    }

    # Get the image as a URL (high-quality image download)
    url = image.getThumbURL(vis_params)
    
    # Fetch the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the response content to an image
        img_io = BytesIO(response.content)
        return img_io
    else:
        raise Exception("Error: Unable to fetch satellite image from Google Earth Engine.")

# Route to fetch and display satellite images
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        # User location
        location_name = request.form.get("location")

        # Fetching coordinates for the location
        try:
            lat, lng = get_coordinates(location_name)
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Fetching satellite image from Google Earth Engine
        try:
            image_io = get_google_earth_image(lat, lng)
            return send_file(image_io, mimetype='image/png')
        except Exception as e:
            return f"Error fetching satellite image: {str(e)}"

    # Render form on GET request
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
