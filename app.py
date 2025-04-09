# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model("rice_leaf_disease_model.h5")  # Load the saved model

# # Class labels
# labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# # Streamlit UI
# st.title("🌾 Rice Leaf Disease Classification")
# st.write("Upload an image of a rice leaf, and the model will predict the disease.")

# # Upload image
# uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Load and preprocess the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert image to numpy array
#     image = np.array(image.convert("RGB"))  # Convert RGBA to RGB

#     image = cv2.resize(image, (180, 180))  # Resize to match model input
#     image = image / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make prediction
#     prediction = model.predict(image)
#     predicted_class = np.argmax(prediction)
#     predicted_label = labels_dict[predicted_class]

#     # Display result
#     st.success(f"✅ **Predicted Disease:** {predicted_label}")

# prefinal_one
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model("rice_leaf_disease_model.keras")  # 👈 Updated here

# # Class labels
# labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# # Streamlit UI
# st.title("🌾 Rice Leaf Disease Classification")
# st.write("Upload an image of a rice leaf, and the model will predict the disease.")

# # Upload image
# uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Load and preprocess the image
#     image = Image.open(uploaded_file)
#     # st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.image(image, caption="Uploaded Image", use_container_width=True)


#     # Convert image to numpy array
#     image = np.array(image.convert("RGB"))  # Convert RGBA to RGB

#     image = cv2.resize(image, (180, 180))  # Resize to match model input
#     image = image / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make prediction
#     prediction = model.predict(image)
#     predicted_class = np.argmax(prediction)
#     predicted_label = labels_dict[predicted_class]

#     # Display result
#     st.success(f"✅ **Predicted Disease:** {predicted_label}")


# initial css1

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from tensorflow.keras.models import load_model
# from lottie_helper import load_lottie_url
# from streamlit_lottie import st_lottie

# # ---- Page Config ----
# st.set_page_config(page_title="Rice Leaf Disease Detector 🌿", layout="centered")

# # ---- CSS Styling ----
# st.markdown("""
#     <style>
#         .main {
#             background-color: #f5f5f5;
#         }
#         h1, h3 {
#             color: #2e7d32;
#         }
#         .stButton > button {
#             background-color: #4caf50;
#             color: white;
#             border-radius: 8px;
#             padding: 0.5em 1.5em;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Sidebar ----
# with st.sidebar:
#     st.title("📖 How to Use")
#     st.markdown("""
#     1. Upload a clear image of a rice leaf 🌿.
#     2. Click the **Predict Disease** button.
#     3. Get the predicted disease and confidence score!
#     """)
#     st.markdown("---")
#     st.caption("Made by Rasagna and Manasa💻")

# # ---- Title & Animation ----
# st.title("🌾 Rice Leaf Disease Classification")

# lottie_leaf = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_2LdLki.json")
# st_lottie(lottie_leaf, height=200, key="leaf")

# st.write("Upload an image of a rice leaf, and the model will predict the disease.")

# # ---- Load Model ----
# model = load_model("rice_leaf_disease_model.keras")
# labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# # ---- File Upload ----
# uploaded_file = st.file_uploader("📁 Choose a leaf image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="📷 Uploaded Image", use_container_width=True)

#     if st.button("🔍 Predict Disease"):
#         # Preprocessing
#         image = np.array(image.convert("RGB"))
#         image = cv2.resize(image, (180, 180))
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)

#         # Prediction
#         prediction = model.predict(image)
#         predicted_class = np.argmax(prediction)
#         confidence = np.max(prediction)
#         predicted_label = labels_dict[predicted_class]

#         # Results
#         st.markdown(f"""
#             <div style='padding: 20px; border-radius: 10px; background-color: #e0f7fa; text-align: center;'>
#                 <h3>🌿 Predicted Disease: <span style='color: #d32f2f;'>{predicted_label}</span></h3>
#                 <p>Confidence: <strong>{confidence:.2%}</strong></p>
#             </div>
#         """, unsafe_allow_html=True)

#         # Show all class probabilities
#         st.markdown("### 🔢 Class Probabilities")
#         for idx, prob in enumerate(prediction[0]):
#             st.write(f"**{labels_dict[idx]}**: {prob:.2%}")




# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from tensorflow.keras.models import load_model
# from lottie_helper import load_lottie_url
# from streamlit_lottie import st_lottie

# # ---- Page Config ----
# st.set_page_config(page_title="🌿 Rice Leaf Classifier", layout="centered")

# # ---- Inject Custom CSS ----
# st.markdown("""
#     <style>
#     html, body {
#         background-color: #f0f9f4;
#     }
#     .main {
#         background-color: #ffffff;
#         padding: 2rem;
#         border-radius: 15px;
#         box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
#     }
#     h1 {
#         color: #1b5e20;
#         text-align: center;
#         margin-bottom: 0;
#     }
#     .stButton>button {
#         background-color: #43a047;
#         color: white;
#         font-size: 18px;
#         padding: 0.5em 1.5em;
#         border-radius: 8px;
#         transition: all 0.3s ease-in-out;
#     }
#     .stButton>button:hover {
#         background-color: #2e7d32;
#         color: #fff;
#         transform: scale(1.05);
#     }
#     .prediction-box {
#         padding: 1.5rem;
#         background-color: #e8f5e9;
#         border-left: 10px solid #388e3c;
#         border-radius: 12px;
#         margin-top: 1rem;
#         text-align: center;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Sidebar ----
# with st.sidebar:
#     st.title("📖 How to Use")
#     st.markdown("""
#     1. Upload a **rice leaf image** 🌾  
#     2. Click **Predict Disease** 🔍  
#     3. View result & class probabilities
#     """)
#     st.markdown("---")
#     st.caption("Made with 💚 by Rasagna and Manasa")

# # ---- Header ----
# st.title("🌾 Rice Leaf Disease Detector")

# lottie_leaf = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_2LdLki.json")
# st_lottie(lottie_leaf, height=200)

# st.write("Upload an image of a rice leaf to predict the disease affecting it.")

# # ---- Load Model ----
# model = load_model("rice_leaf_disease_model.keras")
# labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# # ---- File Upload ----
# uploaded_file = st.file_uploader("📁 Choose a leaf image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="📷 Uploaded Image", use_container_width=True)

#     if st.button("🔍 Predict Disease"):
#         image = np.array(image.convert("RGB"))
#         image = cv2.resize(image, (180, 180))
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)

#         prediction = model.predict(image)
#         predicted_class = np.argmax(prediction)
#         confidence = np.max(prediction)
#         predicted_label = labels_dict[predicted_class]

#         # Show prediction
#         st.markdown(f"""
#         <div class="prediction-box">
#             <h3>🌿 Predicted Disease: <span style='color: #2e7d32'>{predicted_label}</span></h3>
#             <p><strong>Confidence:</strong> {confidence:.2%}</p>
#         </div>
#         """, unsafe_allow_html=True)

#         # Show probabilities
#         st.markdown("### 📊 Class Probabilities")
#         for idx, prob in enumerate(prediction[0]):
#             st.write(f"**{labels_dict[idx]}**: `{prob:.2%}`")



### cAMERA



import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from lottie_helper import load_lottie_url
from streamlit_lottie import st_lottie

# ---- Page Config ----
st.set_page_config(page_title="Rice Leaf Disease Detector", layout="centered")

# ---- Custom CSS ----
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .main {
            font-family: 'Segoe UI', sans-serif;
            padding-top: 20px;
        }
        h1 {
            color: #388e3c;
        }
        .upload-option, .camera-option {
            background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 15px;
        }
        .predict-button {
            background-color: #4caf50 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f1f8e9;
            text-align: center;
            border: 1px solid #cddc39;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar Instructions ----
with st.sidebar:
    st.title("📘 Guide")
    st.write("Choose how you want to input the image:")
    st.markdown("- 📁 Upload a leaf image\n- 📷 Scan via camera")
    st.markdown("Then click **Predict Disease**.")
    st.markdown("---")
    st.caption("Built by Rasagna & Manasa")

# ---- Title & Animation ----
st.title("🌾 Rice Leaf Disease Classifier")

lottie_leaf = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_2LdLki.json")
st_lottie(lottie_leaf, height=180, key="leaf")

# ---- Load Model ----
model = load_model("rice_leaf_disease_model.keras")
labels_dict = {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Smut"}

# ---- Input Selection ----
st.markdown("### 🌿 Select Input Method")

col1, col2 = st.columns(2)
with col1:
    if st.button("📁 Upload Image", use_container_width=True):
        st.session_state.input_method = "upload"
with col2:
    if st.button("📷 Scan Leaf via Camera", use_container_width=True):
        st.session_state.input_method = "camera"

# Default
if "input_method" not in st.session_state:
    st.session_state.input_method = "upload"

image = None

# ---- Upload or Capture Image ----
if st.session_state.input_method == "upload":
    st.markdown('<div class="upload-option">📤 Upload a rice leaf image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose image file", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif st.session_state.input_method == "camera":
    st.markdown('<div class="camera-option">📸 Capture an image using your webcam</div>', unsafe_allow_html=True)
    camera_image = st.camera_input("Click a clear photo of the leaf")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Scanned Image", use_container_width=True)

# ---- Predict Button ----
if image is not None and st.button("🔍 Predict Disease", type="primary"):
    # Preprocessing
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (180, 180))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = labels_dict[predicted_class]

    # Result
    st.markdown(f"""
        <div class="result-box">
            <h3>🌱 Predicted Disease: <span style='color:#d32f2f;'>{predicted_label}</span></h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    # Class Probabilities
    st.markdown("### 🔢 Class Probabilities")
    for idx, prob in enumerate(prediction[0]):
        st.write(f"**{labels_dict[idx]}**: {prob:.2%}")













# import streamlit as st
# import numpy as np
# import pickle
# import cv2
# import tensorflow as tf
# from PIL import Image

# # Load pickle and models
# with open("ensemble_model_bundle.pkl", "rb") as f:
#     bundle = pickle.load(f)

# model1 = tf.keras.models.load_model(bundle["model1_path"])
# model2 = tf.keras.models.load_model(bundle["model2_path"])
# labels_dict1 = bundle["labels_dict"]

# st.title("Rice Leaf Disease Classifier (Ensemble CNN)")

# uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     resized = cv2.resize(image_np, (180, 180)) / 255.0
#     input_img = np.expand_dims(resized, axis=0)

#     pred1 = model1.predict(input_img)
#     pred2 = model2.predict(input_img)
#     final_pred = (pred1 + pred2) / 2
#     class_idx = np.argmax(final_pred)
#     st.image(image, caption=f"Prediction: {labels_dict1[class_idx]}", use_column_width=True)