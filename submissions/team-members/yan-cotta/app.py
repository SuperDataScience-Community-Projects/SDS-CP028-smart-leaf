import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from pathlib import Path
import time

# Configure the Streamlit page
st.set_page_config(
    page_title="Smart Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .upload-text {
        text-align: center;
        padding: 1em;
    }
    .prediction-box {
        padding: 1.5em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
NUM_CLASSES = 14
CLASS_NAMES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Hispa', 'Rice___Leaf_Blast', 'Rice___Neck_Blast',
    'Wheat___Brown_Rust', 'Wheat___Healthy'
]
# Determine the correct model path relative to this app.py file
# app.py is in submissions/team-members/yan-cotta/
# model is in submissions/team-members/yan-cotta/scripts/outputs/best_leaf_disease_model.pth
MODEL_PATH = Path(__file__).parent / "scripts" / "outputs" / "best_leaf_disease_model.pth"

# --- Model Definition ---
class LeafDiseaseResNet(nn.Module):
    """ResNet18 model for leaf disease classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=False) # Set pretrained=False as we load custom weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # No need to freeze layers here as we are loading a fully trained model state

    def forward(self, x):
        return self.model(x)

# --- Image Transformations ---
def get_transform() -> transforms.Compose:
    """Returns image transformations for validation/inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Model Loading ---
@st.cache_resource # Cache the model loading for efficiency
def load_model_cached(model_path, num_classes, device):
    """Loads the pre-trained model."""
    model = LeafDiseaseResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Streamlit UI ---
def run_app(model, device, transform):
    """Creates the Streamlit user interface."""
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### Smart Leaf Disease Classifier 🌿
        
        This application uses deep learning to detect diseases in:
        - � Corn
        - 🥔 Potato
        - 🌾 Rice
        - 🌾 Wheat
        
        Upload a clear image of a leaf to get started!
        """)
        
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Upload a leaf image
        2. Wait for analysis
        3. Review predictions
        
        For best results:
        - Use well-lit images
        - Center the leaf in the frame
        - Avoid blurry photos
        """)

    # Main content
    st.title("Smart Leaf Disease Classifier 🌿")
    st.write("Upload an image of a plant leaf to identify potential diseases.")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is None:
            st.markdown("""
            <div class="upload-text">
                <h3>👆 Upload a leaf image to get started</h3>
                <p>Supported formats: JPG, JPEG, PNG</p>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Display the image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Analyzing image..."):
                # Add a small delay to show the spinner
                time.sleep(0.5)
                
                # Preprocess the image
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_idx = torch.argmax(probabilities).item()
                    predicted_class_name = CLASS_NAMES[predicted_idx]
                    confidence = probabilities[predicted_idx].item()

                # Show results in a nice format
                st.markdown("""
                <div class="prediction-box" style="background-color: #f0f2f6;">
                    <h2>Analysis Results</h2>
                """, unsafe_allow_html=True)
                
                st.success(f"🎯 Predicted Condition: {predicted_class_name.replace('___', ' - ')}")
                st.info(f"📊 Confidence: {confidence:.2%}")

                # Show top 3 predictions
                st.markdown("### Top 3 Possibilities:")
                top_3_idx = torch.topk(probabilities, 3).indices
                for idx in top_3_idx:
                    class_name = CLASS_NAMES[idx].replace('___', ' - ')
                    prob = probabilities[idx].item()
                    st.progress(prob)
                    st.write(f"{class_name}: {prob:.2%}")
                # Display probabilities for all classes
            st.subheader("Prediction Probabilities:")
            probs_df = {CLASS_NAMES[i]: f"{probabilities[i].item():.2%}" for i in range(NUM_CLASSES)}
            st.table(probs_df)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
            # Log the error for debugging if needed
            # print(f"Error: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    current_device = torch.device('cpu') # Use CPU for inference
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        st.stop()

    try:
        loaded_model = load_model_cached(MODEL_PATH, NUM_CLASSES, current_device)
        image_transform = get_transform()
        run_app(loaded_model, current_device, image_transform)
    except Exception as e:
        st.error(f"Failed to load the model or run the app: {e}")
        # Log the error for debugging if needed
        # print(f"Startup Error: {e}")
