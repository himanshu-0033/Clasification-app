import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = './dataset' # Path to your dataset folder containing /dog, /cat, /human, /pig
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
USE_TRANSFER_LEARNING = True # Set to False to use the custom CNN from scratch

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_data(dataset_dir, use_transfer_learning=True):
    # Data Augmentation (Rotation, Flipping, Zoom) & Normalization
    if use_transfer_learning:
        # Use MobileNetV2 preprocessing function (scales pixels between -1 and 1)
        preprocessing_function = mobilenet_preprocess
        rescale = None
    else:
        # Normal Custom CNN: scale pixels between 0 and 1
        preprocessing_function = None
        rescale = 1./255
        
    datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # 80/20 train/test split
    )

    # Train Generator (80%)
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation/Test Generator (20%)
    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False # Keep false to align predictions with true labels for confusion matrix
    )
    
    return train_generator, val_generator

# ==========================================
# MODEL BUILDING
# ==========================================
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4):
    """Basic Custom CNN architecture as requested"""
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting
        Dense(num_classes, activation='softmax') # Output layer with Softmax
    ])
    return model

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4):
    """Transfer learning using MobileNetV2 for significantly better accuracy"""
    # Load the MobileNetV2 base model 
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), # Flatten equivalent for complex base models
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ==========================================
# VISUALIZATION & EVALUATION
# ==========================================
def plot_history(history):
    """Plot accuracy and loss curves"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    # plt.show()

def evaluate_model(model, val_generator):
    """Generate confusion matrix and classification report"""
    print("\n--- Evaluating Model ---")
    val_generator.reset()
    
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())
    
    # Predict probabilities and get highest probability class
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    # plt.show()

# ==========================================
# PREDICTION
# ==========================================
def predict_image(img_path, model, class_indices, use_transfer_learning=True):
    """Predict a single new image"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Preprocess identically to training
    if use_transfer_learning:
        img_array = mobilenet_preprocess(img_array)
    else:
        img_array = img_array / 255.0
        
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    
    # Inverse map indices to class labels
    labels = {v: k for k, v in class_indices.items()}
    predicted_label = labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    print(f"\nPrediction for {img_path}:")
    print(f"Class: {predicted_label} | Confidence: {confidence:.2f}%")
    return predicted_label, confidence

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == '__main__':
    # Make sure dataset directory exists, even if empty
    if not os.path.exists(DATASET_DIR):
        print(f"Directory {DATASET_DIR} not found. Creating placeholder directories...")
        for category in ['dog', 'cat', 'human', 'pig']:
            os.makedirs(os.path.join(DATASET_DIR, category), exist_ok=True)
        print(f"Action required: Please place images in {DATASET_DIR} before running.")
        exit(0)
            
    print("Loading datasets...")
    train_gen, val_gen = load_data(DATASET_DIR, use_transfer_learning=USE_TRANSFER_LEARNING)
    
    # Basic check to avoid errors if there are no images yet
    if train_gen.samples == 0:
        print("\n[!] WARNING: No images found in dataset directories.")
        print(f"Please add your images to {DATASET_DIR}/dog, {DATASET_DIR}/cat, etc.")
        print("Exiting...")
        exit(0)

    print("\nBuilding model...")
    if USE_TRANSFER_LEARNING:
        print("Using MobileNetV2 Transfer Learning Architecture...")
        model = build_transfer_learning_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=4)
    else:
        print("Using Custom CNN Architecture...")
        model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=4)
        
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    print("\nStarting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    
    print("\nSaving Model...")
    model.save('animal_classifier_model.h5')
    
    print("\nPlotting Training Curves...")
    plot_history(history)
    
    print("\nEvaluating Model (Test Data)...")
    evaluate_model(model, val_gen)
    
    print("\nDone! To predict a new image, use predict_image(img_path, model, train_gen.class_indices)")
    # Uncomment to test prediction:
    # predict_image("path/to/test_image.jpg", model, train_gen.class_indices, USE_TRANSFER_LEARNING)
