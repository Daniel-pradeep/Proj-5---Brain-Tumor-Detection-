import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --- Load Model ---
model = load_model("brain_tumor_model.h5")
IMG_SIZE = 128  # Must match training image size

# --- Label Encoder Classes (same order used in training) ---
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Update based on your dataset folder names

# --- Streamlit App ---
st.title("🧠 Brain Tumor MRI Image Classification")
st.write("Upload an MRI image and the model will predict the tumor class.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Remove alpha channel
    elif img_array.shape[-1] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(f"🎯 Predicted Tumor Class: **{predicted_class.upper()}**")
