import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import torch


class_names = np.load('class_names.npy', allow_pickle=True).tolist()

# --- Configuration ---
#MODEL_PATH = 'efficientnet_best_model.keras'  # Path to your saved model
MODEL_PATH = 'best_model.keras'  # Path to your saved model

IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Load Model and Class Names ---
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    # You should have a saved class_names list; if not, manually specify here
    # For example:
    # class_names = ['Corn_Common_Rust', 'Corn_Healthy', ...]
    # Or load from a file, e.g., np.load('class_names.npy')
    #class_names = list(model.class_names) if hasattr(model, "class_names") else ["Class_1", "Class_2", "Class_3"]
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]  
    return model, class_names

model, class_names = load_model_and_classes()

st.title("Crop Disease Image Classifier")
st.write("Upload an image of a crop leaf and the model will predict the crop and disease (if any).")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose a crop leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    # Do NOT normalize if your model includes Rescaling(1./255) as first layer!

    # --- Predict ---
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions, axis=1)[0]
    pred_class_name = class_names[pred_class] if len(class_names) > pred_class else f"Class {pred_class}"

    # --- Display result ---
    st.success(f"**Prediction:** {pred_class_name}")

    # If you want to display class probabilities as well:
    st.subheader("Class Probabilities")
    for idx, prob in enumerate(predictions[0]):
        st.write(f"{class_names[idx] if idx < len(class_names) else idx}: {prob:.4f}")
else:
    st.info("Please upload an image to get a prediction.")

