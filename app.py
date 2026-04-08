import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
MODEL_PATH = 'animal_classifier_model.tflite'

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    MODEL_LOADED = True
    print("TFLite model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print("Warning: Model not found or error loading TFLite model:", e)

class_names = {0: 'cat', 1: 'dog', 2: 'human', 3: 'pig'}

def prepare_image(image_bytes):
    # Convert incoming bytes to a PIL Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to the 224x224 expected by our architecture
    img = img.resize((224, 224))
    
    # Convert image to numpy array format
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch size dimension
    
    # MobileNetV2 preprocessing: scale pixels directly from [0, 255] to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return img_array

@app.route('/')
def home():
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
        
        # Make predictions using TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
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
    app.run(debug=True, port=5000)
