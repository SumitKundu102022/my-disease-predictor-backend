import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf  # Import TensorFlow
from PIL import Image  # Import Pillow (PIL)
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask_cors import CORS
from config import MODEL_PATH,UPLOAD_FOLDER,ALLOWED_EXTENSIONS  # Import the model path from config




app = Flask(__name__)
CORS(app,origins="https://mediscanai-eight.vercel.app")  # Enable CORS for all routes

# Load the model globally on application startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None # Set to None to indicate model loading failure
    
    
# Load class names
# IMPORTANT: Ensure you have a 'class_names.txt' file in your 'models' directory
# (or wherever your MODEL_PATH is pointing if it's a directory)
# This file should contain one class name per line, in the same order as your model's output.
class_names = []
try:
    # Adjust this path if your class_names.txt is not directly next to your .h5 file
    class_names_path = os.path.join(os.path.dirname(MODEL_PATH), 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f]
    print(f"Class names loaded: {class_names}")
except Exception as e:
    print(f"Error loading class names: {e}. Please ensure 'class_names.txt' exists and is correctly formatted.")
    # Fallback if class names cannot be loaded (replace with your known classes if needed)
    # class_names = ['healthy', 'disease_a', 'disease_b'] # Fallback example classes
    
# # Configuration
# UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded images
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed image file extensions
# Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# TensorFlow Model Loading (Replace with your actual model and weights)
# Example:
# model = tf.keras.models.load_model('your_model.h5')
# classes = ['healthy', 'disease_a', 'disease_b']  # Example class names

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    """
    Preprocesses the image for prediction using MobileNetV2's specific preprocessing.
    """
    print(f"Attempting to preprocess image: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image opened successfully. Mode: {img.mode}, Size: {img.size}")

        if img.mode != 'RGB':
            img = img.convert('RGB')
            print("Image converted to RGB mode.")

        img = img.resize((224, 224)) # Ensure this matches your model's input size
        print(f"Image resized to: {img.size}")

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # THIS IS THE CRITICAL CHANGE: Use MobileNetV2's specific preprocessing
        img_array = preprocess_input(img_array)
        
        print("Image preprocessed with MobileNetV2 specific preprocessing (scaled to -1 to 1).")
        print(f"Preprocessed image array shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during image preprocessing: {e}")
        return None




# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads and returns disease predictions.
    """
    global model, class_names # Access the global model and class_names

    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server startup logs for errors.'}), 500
    if not class_names:
        return jsonify({'error': 'Class names not loaded. Cannot interpret predictions.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(filepath) # Save the uploaded image temporarily
        except Exception as e:
            print(f"Error saving image: {e}")
            return jsonify({'error': f'Failed to save image: {e}'}), 500

        try:
            img_array = preprocess_image(filepath)
            if img_array is None:
                os.remove(filepath) # Clean up file
                return jsonify({'error': 'Image preprocessing failed'}), 400
            
            # DEBUGGING: Print the input array shape before prediction
            print(f"Input image array shape for prediction: {img_array.shape}")
            
            predictions = model.predict(img_array)
            probabilities = predictions[0] # Get the probabilities for the single image
            
                        # DEBUGGING: Print raw probabilities
            print(f"Raw model output probabilities: {probabilities}")
            # DEBUGGING: Find the predicted class index and name
            predicted_index = np.argmax(probabilities)
            predicted_class_name = class_names[predicted_index] if predicted_index < len(class_names) else "UNKNOWN"
            print(f"Predicted class index: {predicted_index}, Predicted class name: {predicted_class_name}")
            
            
            formatted_results = []
            # Zip probabilities with class names to create the output
            for i, prob in enumerate(probabilities):
                if i < len(class_names): # Ensure index is within bounds of class_names
                    formatted_results.append({
                        'className': class_names[i],
                        'probability': float(prob) # Convert numpy float to Python float
                    })
                else:
                    # Handle cases where model output has more classes than defined
                    formatted_results.append({
                        'className': f'unknown_class_{i}',
                        'probability': float(prob)
                    })

            os.remove(filepath) # Delete the image file after prediction
            return jsonify({'predictions': formatted_results}), 200

        except Exception as e:
            print(f"Error during prediction: {e}")
            if os.path.exists(filepath):
                os.remove(filepath) # Ensure file is cleaned up even on prediction error
            return jsonify({'error': f'Prediction error: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# if __name__ == '__main__':
#     # Ensure the upload folder exists
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(debug=True, host='0.0.0.0') # Run Flask app
