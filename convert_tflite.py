import tensorflow as tf
import os

model_path = "animal_classifier_model.h5"
if not os.path.exists(model_path):
    print("Model not found! Please ensure animal_classifier_model.h5 exists.")
    exit(1)

print("Loading Keras model...")
model = tf.keras.models.load_model(model_path)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For maximum compatibility and minimum size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = "animal_classifier_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Successfully converted and saved to {tflite_path}")
