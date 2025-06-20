import os
import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Smart Leaf - Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Define class labels
CLASS_NAMES = [
    'Corn (maize) Common rust',
    'Corn (maize) Northern Leaf Blight',
    'Corn (maize) healthy',
    'Potato Early blight',
    'Potato Late blight',
    'Potato healthy',
    'Rice Brown spot',
    'Rice Leaf blast',
    'Rice healthy',
    'Wheat Brown rust',
    'Wheat Yellow rust',
    'Wheat healthy',
    'Tomato Early blight',
    'Tomato healthy'
]

# Model descriptions for the info boxes
MODEL_DESCRIPTIONS = {
    "CNN from Scratch": """
        🔨 A custom Convolutional Neural Network built from scratch.
        - Architecture: Multiple Conv2D and MaxPooling2D layers
        - Training: Trained directly on the crop disease dataset
        - Best for: Understanding baseline performance
        """,
    "EfficientNetB0 (Fine-tuned)": """
        🚀 Transfer learning model based on EfficientNetB0.
        - Architecture: Pre-trained EfficientNetB0 + custom top layers
        - Training: Fine-tuned on crop disease dataset
        - Best for: High accuracy predictions
        """
}

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-text {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .model-select {
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Smart Leaf")
st.subheader("Crop Disease Detection using Deep Learning")
st.markdown("""
This application uses deep learning to detect diseases in crop leaves.
Upload an image of a crop leaf, select a model, and get instant predictions!
""")

# Model loading with error handling
@st.cache_resource
def load_model_cache(model_path):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, model_path)
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def image_preprocessing(image):
    try:
        new_image = Image.open(image)
        new_image = new_image.resize((224, 224))
        new_image = img_to_array(new_image)
        new_image = np.expand_dims(new_image, axis=0)
        new_image = new_image/255.0
        return new_image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Model selection
st.markdown("### 🤖 Select Model")
model_choice = st.radio(
    "Choose a model for prediction:",
    ["CNN from Scratch", "EfficientNetB0 (Fine-tuned)"],
    horizontal=True
)

# Show model description
st.info(MODEL_DESCRIPTIONS[model_choice])

# Load selected model
model_file = "model.keras" if model_choice == "CNN from Scratch" else "efficientnetb0.keras"
model = load_model_cache(model_file)

# Main interface
st.markdown("### 📸 Upload a Crop Leaf Image")
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    try:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with col2:
            if st.button("Predict Disease"):
                with st.spinner(f"Analyzing image using {model_choice}..."):
                    # Preprocess the image
                    processed_image = image_preprocessing(image_file)
                    
                    if processed_image is not None and model is not None:
                        # Get prediction
                        prediction = model.predict(processed_image)
                        predicted_class = np.argmax(prediction)
                        confidence = float(prediction[0][predicted_class])
                        
                        # Display results
                        st.markdown("### Results")
                        result = CLASS_NAMES[predicted_class]
                        
                        # Color code based on health status
                        if "healthy" in result.lower():
                            st.success(f"Prediction: {result}")
                        else:
                            st.error(f"Prediction: {result}")
                        
                        # Display confidence
                        st.progress(confidence)
                        st.info(f"Confidence: {confidence*100:.2f}%")
                        
                        # Additional information
                        if "healthy" not in result.lower():
                            st.warning("⚠️ Please consult with an agricultural expert for treatment options.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
