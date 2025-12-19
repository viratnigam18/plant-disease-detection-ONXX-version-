import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# load class names (same order as training) 
class_names = sorted(os.listdir("plant_disease_data"))

# Load ONNX model 
session = ort.InferenceSession("leaf_disease_model.onnx")
input_name = session.get_inputs()[0].name

# Load and preprocess image
img = Image.open("OIP.jpeg").resize((128, 128))
img = np.array(img).astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Run inference
output = session.run(None, {input_name: img})[0]  # shape: (1, num_classes)

predicted_index = np.argmax(output[0])
predicted_class = class_names[predicted_index]
confidence = output[0][predicted_index] * 100

print("Predicted class :", predicted_class)
print(f"Confidence      : {confidence:.2f}%")
