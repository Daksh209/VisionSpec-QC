import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("../models/model.h5")

# Class names
class_names = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# Image path
img_path = input("Enter image path: ")

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get predicted class
predicted_class = class_names[np.argmax(prediction)]

# Confidence
confidence = np.max(prediction)

print(f"\nPredicted Defect: {predicted_class}")
print(f"Confidence: {confidence:.4f}")