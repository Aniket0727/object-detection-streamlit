import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# Loading trained model
MODEL_PATH = "https://github.com/aniket0727/object-detection-streamlit/releases/download/v1.0/model.h5"
CLASS_NAMES_PATH = "https://github.com/aniket0727/object-detection-streamlit/releases/download/v1.0/class_names.json"



model = tf.keras.models.load_model(MODEL_PATH)


# Loading class names from JSON
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
    confidence = np.max(predictions) * 100
    
    # Converts it from a fraction (0–1) to a percentage.
    # 0.90 * 100 = 90%.
    return predicted_class_name, confidence


# Streamlit UI
st.title("Object Detection")
st.write("Upload an image — the model will classify it into one of the trained categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    label, confidence = predict_image(image)

    st.success(f" Detected: **{label}** ({confidence:.2f}% confidence)")





