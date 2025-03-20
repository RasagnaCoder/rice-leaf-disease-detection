import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("rice_leaf_disease_model.h5")  # Load the saved model

# Class labels
labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# Streamlit UI
st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf, and the model will predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image = np.array(image.convert("RGB"))  # Convert RGBA to RGB

    image = cv2.resize(image, (180, 180))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = labels_dict[predicted_class]

    # Display result
    st.success(f"✅ **Predicted Disease:** {predicted_label}")
