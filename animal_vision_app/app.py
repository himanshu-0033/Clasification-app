import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# We look for the model in the parent directory where you'll run the training script
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'animal_classifier_model.h5')

# Attempt to load the model. If it hasn't been trained yet, we handle gracefully.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    MODEL_LOADED = True
    print(f"OK: Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"Warning: Model not found at {MODEL_PATH}. Have you run 'python model_classifier.py' to train it yet?")

# Alphabetical order assigned by Keras ImageDataGenerator automatically
class_names = {0: 'cat', 1: 'dog', 2: 'human', 3: 'pig'} 

def prepare_image(image_bytes):
    """Decodes, resizes, and preprocesses the image for MobileNetV2"""
    # Convert incoming bytes to a PIL Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to the 224x224 expected by our architecture
    img = img.resize((224, 224))
    
    # Convert image to numpy array format
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch size dimension
    
    # Apply exactly the same preprocessing as in training
    processed_image = preprocess_input(img_array)
    return processed_image

@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'System offline: The AI model is not yet trained or found.'}), 500
        
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received.'}), 400
        
    try:
        # Extract base64 image data string
        image_data = data['image'].split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        
        # Preprocess and prepare for neural network
        processed_image = prepare_image(image_bytes)
        
        # Get raw probabilities
        predictions = model.predict(processed_image)
        
        # Extract best class match
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx]) * 100
        predicted_label = class_names.get(predicted_class_idx, "Unknown")
        
        # Return success payload
        return jsonify({
            'success': True,
            'prediction': predicted_label.capitalize(),
            'confidence': f"{confidence:.1f}%"
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    # Running on 5001 to ensure it doesn't conflict with your existing port 5000 app
    print("\n=============================================")
    print("Starting Animal Vision Web Application...")
    print("Accessible at: http://127.0.0.1:5001")
    print("=============================================\n")
    app.run(debug=True, port=5001)
