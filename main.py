import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import urllib.request

st.title("Object Detection App")

# GitHub Release download URLs
MODEL_URL = "https://github.com/aniket0727/object-detection-streamlit/releases/download/v1.0/model.h5"
CLASS_URL = "https://github.com/aniket0727/object-detection-streamlit/releases/download/v1.0/class_names.json"

# Local file paths
MODEL_PATH = "model.h5"
CLASS_NAMES_PATH = "class_names.json"

# Download model file if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... Please wait ⏳")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.write("Model downloaded successfully! ✅")

# Download class names file if it doesn't exist
if not os.path.exists(CLASS_NAMES_PATH):
    st.write("Downloading class names... ⏳")
    urllib.request.urlretrieve(CLASS_URL, CLASS_NAMES_PATH)
    st.write("Class names downloaded successfully! ✅")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (128, 128)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(predictions) * 100)
    return predicted_class_name, confidence

st.header("Upload an Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300, caption="Uploaded Image")

    label, confidence = predict_image(image)

    st.success(f"Detected: **{label}** ({confidence:.2f}% confidence)")
