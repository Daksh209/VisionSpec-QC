import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("../models/model.h5", compile=False)

# Force build model
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = model(dummy_input)

# Class labels
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
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print(f"\nPredicted Defect: {predicted_class_name}")
print(f"Confidence: {confidence:.4f}")

# Get MobileNet base model
base_model = model.layers[0]

# Last convolution layer
last_conv_layer = base_model.get_layer("Conv_1")

# Create grad model
grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, base_model.output]
)

# Gradient computation
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    tape.watch(conv_outputs)

    pooled_output = tf.reduce_mean(predictions, axis=(1, 2))

    class_channel = pooled_output[:, predicted_class_index]

# Compute gradients
grads = tape.gradient(class_channel, conv_outputs)

# Handle gradient issue
if grads is None:
    raise ValueError("Gradients are None. Model graph not connected properly.")

# Mean gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Feature maps
conv_outputs = conv_outputs[0]

# Create heatmap
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize
heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

# Load original image
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, (224, 224))

# Resize heatmap
heatmap = cv2.resize(heatmap, (224, 224))

# Apply colormap
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay
superimposed_img = cv2.addWeighted(img_cv, 0.4, heatmap, 0.8, 0)

# Save
output_path = "../static/gradcam_result.jpg"
cv2.imwrite(output_path, superimposed_img)

print(f"\nGrad-CAM saved at: {output_path}")

# Show result
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Grad-CAM: {predicted_class_name}")
plt.show()